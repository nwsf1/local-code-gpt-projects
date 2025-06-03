# File: fine_tuning_gptneo_lora/prepare_code_dataset.py
# Purpose: Convert a folder of Python files into a JSONL dataset for fine-tuning.

import os
import json

# ─────────────────────────────────────────────────────────────────────────────
# 1. USER‐EDIT THIS PATH: Folder containing your .py code files
# ─────────────────────────────────────────────────────────────────────────────
CODE_FOLDER = r"G:\learning ai\code_corpus"
OUTPUT_JSONL = r"G:\learning ai\local-code-gpt-projects\fine_tuning_gptneo_lora\data\processed\code_dataset.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# 2. ENSURE THE OUTPUT PATH EXISTS
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. WALK CODE_FOLDER AND WRITE JSONL
# ─────────────────────────────────────────────────────────────────────────────
with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
    for root, _, files in os.walk(CODE_FOLDER):
        for fname in files:
            if fname.endswith(".py"):
                file_path = os.path.join(root, fname)
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                record = {"text": code}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"[INFO] Wrote JSONL dataset to {OUTPUT_JSONL}")
