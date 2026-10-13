"""
run_finetune.py
---------------
Kaggle batch job: fine-tune BrainGPT (Mistral-7B + LoRA) on Kaggle's GPU.

It runs finetune_braingpt.py, which is uploaded next to the training pairs in
a private Kaggle dataset. The trained adapter is written to /kaggle/working,
which Kaggle keeps as the job's output.
"""

import glob
import os
import subprocess
import sys

#one GPU only: the model fits, and it keeps Trainer away from multi-GPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes", "peft"],
               check=True)

#the dataset folder name depends on the dataset slug, so look for the files
script = glob.glob("/kaggle/input/**/finetune_braingpt.py", recursive=True)[0]
data = glob.glob("/kaggle/input/**/training_pairs_v3.jsonl", recursive=True)[0]
print(f"script: {script}\ndata: {data}", flush=True)

subprocess.run(["nvidia-smi"], check=False)

subprocess.run([
    sys.executable, script,
    "--model", "mistral",
    "--data", data,
    "--output", "/kaggle/working/braingpt-mistral-lora-v3",
    "--epochs", "1",
    "--max-length", "1792",
], check=True)
