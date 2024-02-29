import argparse
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,TrainingArguments,AutoConfig
from datasets import Dataset
import torch
import logging
import os
from peft import LoraConfig, TaskType,get_peft_model,prepare_model_for_kbit_training
import pandas as pd
import math
import bitsandbytes as bnb
import transformers
from typing import Dict
from typing import List, Optional
from accelerate import Accelerator
import numpy as np
import random
from datetime import datetime
from clearml import Task
import hydra
from omegaconf import DictConfig, OmegaConf


from utils import get_logger
logger = get_logger("finetune", "info")

os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid warnings about parallelism in tokenizers

SUPPORTED_FLASH_MODELS = ["llama", "mistral", "falcon","mixtral","opt"]
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    custom_tokens:Optional[List[str]]=None,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    if len(list(special_tokens_dict.keys())) >0 or custom_tokens is not None:
        logger.info("Resizing tokenizer and embedding...")
        logger.info("Special tokens dict: %s", special_tokens_dict)
        logger.info("Custom tokens: %s", custom_tokens)
    else:
        return False
    num_new_tokens = len(list(special_tokens_dict.keys())) + (0 if custom_tokens is None else len(custom_tokens))
    logger.info("Number of new tokens: %d", num_new_tokens)
    if len(list(special_tokens_dict.keys())) > 0:
        tokenizer.add_special_tokens(special_tokens_dict)
    if custom_tokens is not None:
        tokenizer.add_tokens(custom_tokens,special_tokens=True)

    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
    return True


def find_all_linear_names(args, model,add_lm_head=True):
    cls = bnb.nn.Linear4bit if args.use_int4 else (bnb.nn.Linear8bitLt if args.use_int8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if add_lm_head and not "lm_head" in lora_module_names:
        logger.info("Adding lm_head to lora_module_names")
        lora_module_names.add("lm_head")

    return list(lora_module_names)

def get_config(args):
    config_kwargs = {
        "trust_remote_code": True if args.train["trust_remote_code"] else None,
        "token":args.model["token"]
    }
    config = AutoConfig.from_pretrained(args.model["model_name"], **config_kwargs)

    config.use_cache = False
    if not args.train["gradient_checkpointing"]:
        logger.info("Not using gradient checkpointing")
        config.gradient_checkpointing = False
    else:
        logger.info("Using gradient checkpointing")
        config.gradient_checkpointing = True

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.model["block_size"] > orig_ctx_len and args.rope_scale is None:
        scaling_factor = float(math.ceil(args.model["block_size"] / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        logger.info("Scaling context length by %f", scaling_factor)
    elif args.rope_scale is not None:
        scaling_factor = float(math.ceil(args.rope_scale))
        logger.info("Scaling context length by %f", scaling_factor)
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    else:
        logger.info("Not scaling context length")
    return config

    
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args : DictConfig):
    dt = (
        str(datetime.now().replace(microsecond=0))
        .replace(" ", "_")
        .replace(":", "_")
        .replace("-", "_")
    )

    if not args.clearml["disable_clearml"]:
        task = Task.init(project_name=args.clearml["project_name"], task_name=args.clearml["experiment_name"] if args.clearml["experiment_name"] else f"llama-7b_{dt}")
        
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:', torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)

    seed_all(args.seed)

    if args.peft["lora_alpha"] is None:
        args.peft["lora_rank"] = args.peft["lora_rank"] * 2
        logger.info("Lora alpha set to None... Setting lora_alpha to %d", args.args.peft["lora_alpha"])

    # replace_llama_attn(use_full=False)

    if args.model["token"] is None:
        access_token = os.getenv("HF_TOKEN", "")
    else:
        access_token = args.model["token"]
    
    if args.model["token"] is None:
        args.model["token"] = access_token

    config = get_config(args)
    config_dict = config.to_dict()
    model_type = config_dict["model_type"]


    use_flash_attention = False

    if not args.train["disable_flash_attention"] and  model_type not in SUPPORTED_FLASH_MODELS:
        logger.info("Model is not llama, mistral, or falcon disabling flash attention...")
    elif args.train["disable_flash_attention"] and model_type in SUPPORTED_FLASH_MODELS:
        logger.info("Model is llama, mistral or falcon could be using flash attention...")
    elif not args.train["disable_flash_attention"]:
        logger.info("Using flash attention...")
        use_flash_attention = True


    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "GPT_finetuning"

    if args.model["split_model"]:
        logger.info("Splitting the model across all available devices...")
        kwargs = {"device_map":"auto"}
    else:
        kwargs = {"device_map":None}

    tokenizer = AutoTokenizer.from_pretrained(args.model["model_name"], token=access_token,trust_remote_code=args.train["trust_remote_code"],add_eos_token=args.add_eos_token,add_bos_token=args.add_bos_token,use_fast=True)
    custom_tokens = None
    if args.custom_tokens is not None:
        logger.info("Adding custom tokens to tokenizer...")
        with open(args.custom_tokens,"r") as f:
            custom_tokens = f.readlines()
        custom_tokens = [token.strip() for token in custom_tokens]




    #THIS IS A HACK TO GET THE PAD TOKEN ID NOT TO BE EOS
    #good one for LLama is 18610
    if args.pad_token_id is not None:
        logger.info("Using pad token id %d", args.pad_token_id)
        tokenizer.pad_token_id = args.pad_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(args.pad_token_id)


    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN


    block_size = args.model["block_size"]
    logger.info("Using a block size of %d", block_size)

    if args.quant["use_int4"]:
        logger.info("Using int4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if not args.model["split_model"]:
            device_index = Accelerator().process_index
            device_map = {"": device_index}
            kwargs["device_map"] = device_map
        optimizer = "adamw_bnb_8bit"
        args.use_int8 = False
    elif args.quant["use_int8"]:
        logger.info("Using int8 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        optimizer = "adamw_bnb_8bit"
    else:
        logger.info("Using no quantization")
        bnb_config = None
        optimizer = "adamw_torch"

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model["model_name"], token=access_token, quantization_config=bnb_config,trust_remote_code=args.train["trust_remote_code"],torch_dtype=torch_dtype, config=config, use_flash_attention_2=use_flash_attention, **kwargs)
    added_tokens = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
        custom_tokens=custom_tokens
    )

    if not args.peft["disable_lora"] and args.peft["all_linear"]:
        target_modules = find_all_linear_names(args, model)
        logger.info("Using LORA on all linear layers: %s", target_modules)
        if added_tokens:
            target_modules.pop(target_modules.index("lm_head"))
            logger.info("Removing lm_head from target modules, will use in modules_to_save")
    elif not args.peft["disable_lora"]:
        target_modules = None
        logger.info("Using LORA on default layers")


    if not args.peft["disable_lora"]:
        if args.peft["long_lora"]:
            logger.info("Using long lora settings...")
            modules_to_save = ["embed_tokens","input_layernorm","post_attention_layernorm","norm"]

            if added_tokens:
                logger.info("Adding lm_head to modules_to_save")
                modules_to_save.append("lm_head")
        elif added_tokens:
            modules_to_save = modules_to_save = ["embed_tokens","lm_head"]
        else:
            modules_to_save = None
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.peft["lora_rank"], lora_alpha=args.peft["lora_alpha"], lora_dropout=args.peft["lora_dropout"],target_modules=target_modules,modules_to_save=modules_to_save
        )
        logger.info("Using LORA...")
        if args.quant["use_int4"] or args.quant["use_int8"]:
            logger.info("Preparing model for kbit training...")
            model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=True if args.train["gradient_checkpointing"] else False)

        logger.info("Getting PEFT model...")
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
    else:
        logger.info("Using Full Finetuning")


    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=args.train["output_dir"],
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=args.train["epochs"],
        eval_steps=args.train["eval_steps"],
        save_steps=args.train["save_steps"],
        logging_steps=args.train["log_steps"],
        per_device_train_batch_size=args.train["batch_size"],
        per_device_eval_batch_size=args.train["batch_size"]*2,
        optim=optimizer,
        learning_rate=args.train["learning_rate"],
        lr_scheduler_type=args.train["lr_scheduler_type"],
        warmup_steps=args.train["warmup_steps"],
        gradient_accumulation_steps=args.train["gradient_accumulation_steps"],
        gradient_checkpointing=args.train["gradient_checkpointing"],
        weight_decay=args.train["weight_decay"],
        report_to="tensorboard",
        load_best_model_at_end=True,
        save_total_limit=args.train["save_limit"],
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
    )

    train_df = pd.read_csv(args.train_file)
    if args.train_dataset_ratio < 1.0:
        train_df = train_df.sample(frac=args.train_dataset_ratio,random_state=args.seed)
    train_dataset = Dataset.from_pandas(train_df)
    validation_df = pd.read_csv(args.validation_file)
    if args.validation_dataset_ratio < 1.0:
        validation_df = validation_df.sample(frac=args.validation_dataset_ratio,random_state=args.seed)
    validation_dataset = Dataset.from_pandas(validation_df)

    if args.completion_only:
        logger.info("Using completion only loss...")
        logger.warning("Make sure to manually set this value in the code to a list of ids")
        response_template = None
        assert response_template is not None, "Response template must be set"
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,tokenizer=tokenizer
        )
        packing = False
    else:
        data_collator = None
        packing = None

    # get trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        dataset_text_field=args.sft["dataset_text_field"],
        max_seq_length=block_size,
        tokenizer=tokenizer,
        data_collator=data_collator,
        packing=packing,
    )

    # train
    trainer.train()


if __name__ == "__main__":
    main()