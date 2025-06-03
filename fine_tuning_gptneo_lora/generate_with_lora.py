# File: fine_tuning_gptneo_lora/generate_with_lora.py
# Purpose: Load quantized GPT-Neo-2.7B + LoRA adapters to generate code.
#
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 1. CUDA CONFIG
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Aborting.")

# 2. PATHS
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_ID = "EleutherAI/gpt-neo-2.7B"
LORA_PATH = os.path.join(BASE_DIR, "gptneo-lora-output")

# 3. TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# 4. LOAD BASE MODEL (4-BIT)
print("[INFO] Loading base model in 4-bit …")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
base_model.resize_token_embeddings(len(tokenizer))

# 5. LOAD LoRA ADAPTERS
print("[INFO] Applying LoRA adapters …")
model = PeftModel.from_pretrained(base_model, LORA_PATH, device_map="auto")

# 6. GENERATION FUNCTION
def generate_code(prompt: str, max_new_tokens=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.02,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(output_ids[0][ inputs["input_ids"].shape[-1] : ], skip_special_tokens=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python generate_with_lora.py \"<Your code-generation prompt>\"")
        sys.exit(1)
    prompt_text = sys.argv[1]
    print("[INFO] Generating code…")
    code = generate_code(prompt_text)
    print("\n" + "="*80)
    print(code)
    print("="*80 + "\n")
