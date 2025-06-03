# File: fine_tuning_gptneo_lora/train_neo2_lora.py
# Purpose: Fine-tune GPT-Neo 2.7B on a code dataset using LoRA + 4-bit quantization.
#
# OS: Windows 11 64-bit, RTX 4090 (24 GiB), CUDA 11.7
# Python venv: ../../gptj-env/
# Dependencies: bitsandbytes, transformers, peft, datasets, tokenizers, safetensors
#
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────────────────────────────────────
# 1. PYTORCH MEMORY CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEVICE CHECK
# ─────────────────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Aborting.")
if torch.cuda.device_count() != 1:
    raise EnvironmentError(f"Expected 1 GPU, but found {torch.cuda.device_count()}. Aborting.")
print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PATHS & HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "EleutherAI/gpt-neo-2.7B"
DATA_PATH = "data/processed/code_dataset.jsonl"
OUTPUT_DIR = "gptneo-lora-output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LoRA hyperparameters
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["query_key_value"]

# Training hyperparameters
BATCH_SIZE = 1
GRAD_ACC_STEPS = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 1024

# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# ─────────────────────────────────────────────────────────────────────────────
# 5. LOAD & TOKENIZE DATASET
# ─────────────────────────────────────────────────────────────────────────────
print(f"[INFO] Loading dataset from {DATA_PATH} …")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

print("[INFO] Tokenizing dataset …")
tokenized_dataset = dataset.map(
    tokenize_fn, batched=True, remove_columns=["text"]
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. LOAD GPT-NEO 2.7B IN 4‐BIT MODE & APPLY LoRA
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Loading GPT-Neo 2.7B in 4-bit quantized mode …")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))

print("[INFO] Applying LoRA adapters …")
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES
)
model = get_peft_model(model, peft_config)

# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAINING ARGUMENTS
# ─────────────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    save_steps=500,
    remove_unused_columns=False,
    optim="adamw_torch",
    report_to="none",
    ddp_find_unused_parameters=False,
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. DATA COLLATOR
# ─────────────────────────────────────────────────────────────────────────────
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAIN & SAVE
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Starting training …")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
trainer.train()

print("[INFO] Saving LoRA adapters …")
model.save_pretrained(OUTPUT_DIR)
print(f"[INFO] Fine-tuned model and LoRA weights saved in {OUTPUT_DIR}")
