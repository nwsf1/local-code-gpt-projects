# code_generator

**Purpose**  
A simple script that uses GPT-Neo-125M (FP16) on RTX 4090 to generate “full program” code from a single prompt text file.

## Folder Structure

code_generator/
├── generate_code.py ← Main script to run
├── prompts/
│ └── program_prompt.txt ← Text file containing the instruction
└── generated_program.py ← Output produced by the script

r
Copy
Edit

## Setup & Usage

1. **Activate** your venv:
   ```powershell
   cd "G:\learning ai\local-code-gpt-projects\code_generator"
   ..\gptj-env\Scripts\Activate.ps1
Install dependencies (if not already installed in gptj-env):

powershell
Copy
Edit
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers safetensors
Edit prompts/program_prompt.txt with your desired instructions for the code. For example:

markdown
Copy
Edit
Write a complete Python program named `file_manager.py` that:
1. Recursively traverses a given directory (CLI argument).
2. Lists files larger than 10 MB.
3. If run with `--delete`, deletes those files after listing.
4. Logs actions to `file_manager.log`.
Include all imports and error handling.
Run:

powershell
Copy
Edit
python generate_code.py
Output:

The script prints the generated code to the console.

Saves it as generated_program.py in the same folder.

Troubleshooting
If you see [ERROR] CUDA is not available. Aborting., check your GPU driver and CUDA installation.

If tokenization or model loading fails, ensure transformers>=4.35.0 and safetensors are installed:

powershell
Copy
Edit
pip install --upgrade transformers safetensors
If generated_program.py is missing, verify that prompts/program_prompt.txt is non‐empty and properly formatted.
