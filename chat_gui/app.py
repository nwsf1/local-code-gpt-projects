# File: chat_gui/app.py
# Purpose: ChatGPT-style GUI using Flask + GPT-Neo-2.7B to generate full programs.
#
# OS: Windows 11 64-bit
# GPU: RTX 4090 (24 GiB)
# Python venv: ../gptj-env/
# Dependencies (inside gptj-env):
#   torch ≥2.1.0+cu117, transformers ≥4.35.0, safetensors, flask
#
# Usage:
#   1. Activate venv:
#       cd ../chat_gui
#       ..\gptj-env\Scripts\Activate.ps1
#   2. Run:
#       python app.py
#   3. Open browser at http://127.0.0.1:5000/
#
import os
import torch
from flask import Flask, render_template, request, redirect, url_for
from transformers import GPTNeoForCausalLM, GPTNeoConfig, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# 1. ENVIRONMENT CHECKS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Aborting.")
if torch.cuda.device_count() != 1:
    raise EnvironmentError(f"Expected 1 GPU, but found {torch.cuda.device_count()}. Aborting.")
print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD MODEL & TOKENIZER (ONCE AT STARTUP)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "EleutherAI/gpt-neo-2.7B"
print("[INFO] Loading GPT-Neo-2.7B tokenizer & model (FP16)…")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

config = GPTNeoConfig.from_pretrained(MODEL_ID)
model = GPTNeoForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))
model.eval()
print("[INFO] Model loaded to GPU")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FLASK SETUP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# In-memory history (for display). Each entry: {'user': str, 'bot': str}
conversation_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global conversation_history

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if user_input:
            conversation_history.append({"user": user_input, "bot": None})

            # Use ONLY the raw instruction as prompt (no “User:/Bot:”)
            prompt_text = user_input

            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            ).strip()

            conversation_history[-1]["bot"] = generated_text

        return redirect(url_for("index"))

    return render_template("index.html", history=conversation_history)

@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history
    conversation_history = []
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
