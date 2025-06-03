# fine_tuning_gptneo_lora

**Purpose**  
Fine-tune GPT-Neo-2.7B on a custom code dataset using LoRA + 4-bit quantization so it can generate complete programs on an RTX 4090 (24 GiB).

## Folder Structure

fine_tuning_gptneo_lora/
├── data/
│ └── processed/
│ └── code_dataset.jsonl ← Created by prepare_code_dataset.py
├── prepare_code_dataset.py ← Converts .py files to JSONL
├── train_neo2_lora.py ← LoRA + 4-bit fine-tuning script
├── generate_with_lora.py ← Inference using the fine-tuned LoRA model
├── requirements.txt ← Python dependencies for fine‐tuning
└── README.md ← This documentation

r
Copy
Edit

## Environment & Setup

1. **Activate venv**:
   ```powershell
   cd "G:\learning ai\local-code-gpt-projects\fine_tuning_gptneo_lora"
   ..\..\gptj-env\Scripts\Activate.ps1
Install dependencies:

powershell
Copy
Edit
pip install --upgrade pip
pip install bitsandbytes accelerate peft transformers datasets tokenizers safetensors
Prepare your code corpus:

Put all your .py files under G:\learning ai\code_corpus\ (or another path of your choice).

Edit prepare_code_dataset.py’s CODE_FOLDER to point to that folder.

Run:

powershell
Copy
Edit
python prepare_code_dataset.py
This creates data/processed/code_dataset.jsonl used for training.

Fine-tune GPT-Neo-2.7B:

powershell
Copy
Edit
python train_neo2_lora.py
Training will take ~30–45 minutes for 3 epochs on an RTX 4090.

LoRA adapters are saved into gptneo-lora-output/.

Generate code with your fine-tuned model:

powershell
Copy
Edit
python generate_with_lora.py "Write a Python CLI file_manager.py that …"
The generated code prints to console.

Troubleshooting
If you encounter OutOfMemoryError during from_pretrained(), ensure you installed bitsandbytes correctly and that CUDA 11.7 is active.

If peft is missing, run:

powershell
Copy
Edit
pip install peft
If dataset JSONL is not found, check you ran prepare_code_dataset.py and that code_dataset.jsonl exists.
