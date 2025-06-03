# local-code-gpt-projects

A collection of local AI code‐generation and fine‐tuning examples for GPT‐Neo on an RTX 4090 (24 GiB).

---

## Contents

- [Projects](#projects)  
- [Repository Structure](#repository-structure)  
- [Prerequisites (Global)](#prerequisites-global)  
- [Getting Started](#getting-started)  
- [License](#license)  
- [Contact](#contact)  

---

## Projects

This repository contains **three** distinct, self‐contained subprojects:

1. **`chat_gui/`**  
   A ChatGPT‐style web interface (Flask) running GPT-Neo 2.7B (FP16) that generates full Python programs from a single instruction.

2. **`code_generator/`**  
   A minimal script using GPT-Neo 125M (FP16) to generate complete code files based on a prompt text file.

3. **`fine_tuning_gptneo_lora/`**  
   A LoRA + 4-bit quantization fine-tuning pipeline to train GPT-Neo 2.7B on your own code corpus, plus an inference script to generate code from the fine-tuned adapters.

---

## Repository Structure

```text
local-code-gpt-projects/
├── .github/                         ← (optional) GitHub configuration (workflows, templates, etc.)
├── chat_gui/                        ← Chat GUI project
│   ├── app.py
│   ├── requirements.txt
│   ├── templates/
│   │   └── index.html
│   └── README.md
├── code_generator/                  ← Code generator script
│   ├── generate_code.py
│   ├── prompts/
│   │   └── program_prompt.txt
│   └── README.md
├── fine_tuning_gptneo_lora/         ← LoRA fine-tuning pipeline
│   ├── data/
│   │   └── processed/
│   │       └── .gitkeep
│   ├── prepare_code_dataset.py
│   ├── train_neo2_lora.py
│   ├── generate_with_lora.py
│   ├── requirements.txt
│   └── README.md
└── README.md                        ← This file

Prerequisites (Global)
Operating System

Windows 11 ×64

GPU

NVIDIA RTX 4090 with 24 GiB VRAM

NVIDIA driver ≥ 531.41

CUDA Toolkit 11.7 installed

Python

3.10.10 ×64

Virtual Environment
We assume a global venv located at:

makefile
Copy
Edit
G:\learning ai\gptj-env\
with the following packages installed:

powershell
Copy
Edit
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers safetensors
pip install flask bitsandbytes accelerate peft datasets tokenizers safetensors
Note: Adjust per project if additional packages are required.

Getting Started
To explore any subproject, click its folder in this repo and follow the instructions in that folder’s README.md:

chat_gui/README.md

code_generator/README.md

fine_tuning_gptneo_lora/README.md

Each subproject’s README includes:

Setup steps

Dependency lists

How to run or fine-tune models

Troubleshooting tips

Simply navigate to the folder you want and follow its readme.

License
This code is released under the MIT License.

Contact
For questions, issues, or contributions:

Open an issue on GitHub

Or email: nwsf31@gmail.com

Thank you for exploring the local-code-gpt-projects repository!
Happy coding.
