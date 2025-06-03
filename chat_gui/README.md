# chat_gui

**Purpose**  
A ChatGPT‐style GUI running locally on Windows 11 + RTX 4090 (24 GiB VRAM) using Flask and GPT-Neo 2.7B (FP16).  
It accepts a single “Write a complete Python program that …” instruction and returns a full, runnable `.py` script.

## Folder Structure

chat_gui/
├── app.py ← Flask app (loads GPT‐Neo 2.7B and serves HTML)
├── requirements.txt ← Python dependencies
└── templates/
└── index.html ← HTML template for chat interface

markdown
Copy
Edit

## Environment & Setup

1. **OS**: Windows 11 ×64  
2. **GPU**: NVIDIA RTX 4090 (24 GiB VRAM, CUDA ≥ 11.7)  
3. **Python**: 3.10.10 ×64  
4. **Virtualenv**: Assume you have a venv at `../gptj-env/` with:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers safetensors flask

r
Copy
Edit

## How to Run

```powershell
cd "G:\learning ai\local-code-gpt-projects\chat_gui"
..\gptj-env\Scripts\Activate.ps1
python app.py
Open your browser at http://127.0.0.1:5000/

Type your instruction (e.g. “Write a complete Python program that prints the Fibonacci sequence up to 100”) and click Send

Wait ~5–10 seconds; the bot will reply with a complete Python script.

To clear history, click Reset.

Key Points
Single‐turn raw prompt: The app sends exactly your instruction to GPT-Neo, not “User:/Bot:” history.

Greedy decoding: temperature=0.0, do_sample=False ensures deterministic, coherent code outputs.

FP16 model: GPT-Neo 2.7B in FP16 uses ≈16 GiB VRAM + some overhead (~2 GiB) on RTX 4090.

Troubleshooting
If you see CUDA not available, confirm your driver and CUDA toolkit are properly installed.

If import errors occur, re‐run:

lua
Copy
Edit
..\gptj-env\Scripts\Activate.ps1
pip install --upgrade transformers safetensors flask torch
If the model still generates gibberish or partial code, reduce max_new_tokens or verify temperature=0.0.
