# File: code_generator/generate_code.py
# Purpose: Prompt-driven code generator using GPT-Neo-125M.
#
# OS: Windows 11 ×64
# GPU: RTX 4090 (24 GiB)
# Python venv: ../gptj-env/
#
import os
import sys
import torch
from transformers import AutoTokenizer, GPTNeoForCausalLM

# 1. ENVIRONMENT CHECKS
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available. Aborting.")
    sys.exit(1)
print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")

# 2. LOAD MODEL & TOKENIZER
MODEL_ID = "EleutherAI/gpt-neo-125M"
print("[INFO] Loading GPT-Neo-125M tokenizer & model (FP16)…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = GPTNeoForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))
model.eval()

# 3. READ PROMPT FROM FILE
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "program_prompt.txt")
if not os.path.isfile(PROMPT_FILE):
    print(f"[ERROR] Prompt file not found: {PROMPT_FILE}")
    sys.exit(1)

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_text = f.read().strip()
if not prompt_text:
    print("[ERROR] Prompt file is empty. Aborting.")
    sys.exit(1)

# 4. GENERATE CODE
inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
print("[INFO] Generating code…")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.02,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
generated_code = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
).strip()

# 5. PRINT & SAVE
print("\n" + "="*80 + "\n[GENERATED PROGRAM]\n" + "="*80 + "\n")
print(generated_code)
out_file = os.path.join(os.path.dirname(__file__), "generated_program.py")
with open(out_file, "w", encoding="utf-8") as out_f:
    out_f.write(generated_code)
print(f"\n[INFO] Generated program saved to {out_file}")
