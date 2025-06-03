# local-code-gpt-projects

A collection of local AI code‐generation and fine‐tuning examples for GPT‐Neo on an RTX 4090 (24 GiB).

This repository contains three distinct, self‐contained projects:

1. **chat_gui/**  
   A ChatGPT‐style web interface (Flask) running GPT-Neo 2.7B (FP16) to generate full Python programs from a single instruction.

2. **code_generator/**  
   A minimal script using GPT-Neo 125M (FP16) to generate complete code files based on a prompt text file.

3. **fine_tuning_gptneo_lora/**  
   A LoRA + 4-bit quantization fine-tuning pipeline to train GPT-Neo 2.7B on your own code corpus, along with an inference script to generate code from the fine-tuned adapters.

---

## Repository Structure

local-code-gpt-projects/
├── .github/ ← GitHub configuration (optional)
├── chat_gui/ ← Chat GUI project
├── code_generator/ ← Code generator script
├── fine_tuning_gptneo_lora/ ← LoRA fine‐tuning pipeline
└── README.md ← This file

markdown
Copy
Edit

## Prerequisites (Global)

- **Operating System**: Windows 11 ×64  
- **GPU**: NVIDIA RTX 4090 with 24 GiB VRAM, driver ≥ 531.41, CUDA Toolkit 11.7 installed  
- **Python**: 3.10.10 ×64  
- **Virtual Environment**: We assume a global venv located at `G:\learning ai\gptj-env\` with:

  ```powershell
  pip install --upgrade pip
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
  pip install transformers safetensors
  pip install flask bitsandbytes accelerate peft datasets tokenizers safetensors
Adjust as needed per‐project if additional packages are required.

Getting Started
Browse this repository on GitHub to view each project’s README content:

chat_gui/README.md

code_generator/README.md

fine_tuning_gptneo_lora/README.md

To work on any project, click its folder name and follow the instructions in its README.md.

License
This code is released under the MIT License. See LICENSE (if you choose to add one) for details.
