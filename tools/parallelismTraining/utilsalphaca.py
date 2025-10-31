import os
from enum import Enum

import packaging.version
import torch
import transformers
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig


DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    import json
    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import train_test_split

    # 1. 加载 JSON 格式数据集
    dataset_path = data_args.dataset_name
    if dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")

    # 2. 转换 Alpaca 格式为 messages 对话格式
    def alpaca_to_messages(example):
        system_prompt = "你是一名专业的地理加权模型分析助手，请根据指令准确生成R语言代码或解释。"
        if example.get("input"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{example['instruction']}\n{example['input']}"},
                {"role": "assistant", "content": example["output"]}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]}
            ]
        return {"messages": messages}

    # 转换整个列表
    data_converted = [alpaca_to_messages(d) for d in data_list]

    # 3. 划分训练/验证集（默认 9:1）
    train_data, valid_data = train_test_split(data_converted, test_size=0.1, random_state=42)
    raw_datasets = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(valid_data)
    })

    # 4. 应用 chat_template 处理
    if apply_chat_template:
        def preprocess(samples):
            texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in samples["messages"]]
            return {"content": texts}

        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]

    print(f"Loaded {len(train_data)} training samples and {len(valid_data)} validation samples from {dataset_path}")
    #print("Example conversation:\n", train_data[0]["messages"])

    return train_data, valid_data


def create_and_prepare_model(args, data_args, training_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        if torch.xpu.is_available():
            raise NotImplementedError("XPU hasn't supported unsloth yet")
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=training_args.max_seq_length,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        dtype = quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32

        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": dtype,
        }
        if args.use_flash_attn:
            if torch.xpu.is_available():
                print("XPU hasn't supported flash_attn yet, use eager implementation instead.")
                model_kwargs["attn_implementation"] = "eager"
            else:
                model_kwargs["attn_implementation"] = "flash_attention_2"

        # Only add quantization_config if bnb_config is not None
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    peft_config = None
    chat_template = None
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template

        # make embedding resizing configurable?
        # Transformers 4.46.0+ defaults uses mean_resizing by default, which fails with QLoRA + FSDP because the
        # embedding could be on meta device, therefore, we set mean_resizing=False in that case (i.e. the status quo
        # ante). See https://github.com/huggingface/accelerate/issues/1620.
        uses_transformers_4_46 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.46.0")
        uses_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
        # Check if the model is quantized
        is_quantized = (bnb_config is not None) or (getattr(model, "hf_quantizer", None) is not None)
        if is_quantized and uses_fsdp and uses_transformers_4_46:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
        else:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=training_args.max_seq_length,
        )

    return model, peft_config, tokenizer
