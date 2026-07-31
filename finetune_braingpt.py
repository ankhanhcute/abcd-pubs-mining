"""
finetune_braingpt.py
-------------------
Fine-tune a base model (Llama-2-7b-chat or Mistral-7B-v0.1) with LoRA on IV/DV extraction training pairs produced by format_training_data.py
- LoRA (via peft) freezes the base model and only trains small adapter
  matrices, so it's far cheaper than full fine-tuning -- but on CPU, the
  frozen base model still has to run every forward/backward pass, which is
  the actual bottleneck. See the timing check in Step 0 below before
  committing to a full run.
- Switchable base model: set --model llama2 or --model mistral. Both are
  chat/instruct-tuned already, so the prompt format matters (see
  format_prompt() for each).
- AUTO GPU/CPU DETECTION: this script checks for a CUDA GPU at startup.
  If found, it automatically loads the model in 4-bit (QLoRA) for speed
  and memory efficiency, and uses fp16 mixed precision training. If no
  GPU is found, it falls back to the CPU full-precision path. You do NOT
  need to change any flags when you move from CPU (now) to GPU (once
  access comes through) - just run the same command on the GPU machine.

USAGE
-----
    # Quick timing check on 8 examples, 1 model, 5 steps -- do this first
    python3 finetune_braingpt.py --model llama2 --data training_pairs.jsonl --time-check
 
    # Full run
    python3 finetune_braingpt.py --model llama2 --data training_pairs.jsonl \
        --output ./braingpt-llama2-lora --epochs 3 --batch-size 1
 
    # Same, but Mistral instead
    python3 finetune_braingpt.py --model mistral --data training_pairs.jsonl \
        --output ./braingpt-mistral-lora --epochs 3 --batch-size 1
 
EXPECTED INPUT FORMAT (from format_training_data.py)
------------------------------------------------------
JSONL, one object per line:
    {"prompt": "...paper excerpt + instruction...", "completion": "IV: ...\\nDV: ..."}
    {"prompt": "...", "completion": "No relationship found."}

"""

import argparse
import json
import os
import time

import torch 
from datasets import Datasets
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

GPU_AVAILABLE = torch.cuda.is_available()


#-------1. Model registry - add new base model, nothing need to change-------

MODEL_REGISTRY = {
    "llama2": {
        "hf_id": "meta-llama/Llama-2-7b-chat-hf",
        #Llama 2 chat expected an instruction wrapper
        "prompt_template": "[INST] {instruction} [/INST]",
        
    },
    "mistral": {
        "hf_id": "mistralai/Mistral-7B-v0.1"
        #Mistral-instruct-style wrapper (base v0.1 isn't instruct-tuned),
        #but this keeps format consistent is you swap to Mistral-Instruct later
        "promp_template": "<s>[INST] {instruction} [/INST]"
    },
    "mistralai": {
        "hf_id": "Mistral-7B-Instruct-v0.2",
        "prompt_template": "<s>[INST] {instruction} [/INST]"
    }
    
}
def format_prompt(model_key: str, raw_prompt: str) -> str:
    """Wrap the raw instruction text the base model's expected format"""
    template = MODEL_REGISTRY[model_key]["prompt_template"]
    return template.format(instruction=raw_prompt)
#---------2. Data loading ------------
def load_training_pairs(path: str) -> list[dict]:
    """Load prompt/completion pairs from format_training.py's JSON output"""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_num} of {path}: {e}")
            if "prompt" not in obj or "completion" not in obj:
                raise ValueError(f"Line {line_num} missing 'prompt or 'completion key: {obj}")
            pairs.append(obj)
    if not pairs:
        raise ValueError(f"No training pairs found in {path} - is the file empty?")
    return pairs

def build_dataset(pairs: list[dict], model_key: str, tokenizer, max_length: int = 1024) -> Dataset:
    """Turn prompt/completion pairs into a tokenized HF Dataset for causal LM training."""
    
    def _format_and_tokenize(example):
        full_prompt = format_prompt(model_key, example["prompt"])
        full_text = f"{full_prompt}\n{example['completion']}"
        tokenized = tokenizer(
            full_text, 
            truncation=True, 
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    ds = Dataset.from_list(pairs)
    ds = ds.map(_format_and_tokenize, remove_columns=ds.column_names)
    return ds

#-----3.Model + LoRA Setup-----
def load_model_and_tokenizer(model_key: str):
    hf_id = MODEL_REGISTRY[model_key]["hf_id"]
    print(f"Loading base model: {hf_id} (this downloads ~13-27GB the first time)")
    print(f"Device detected: {'GPU (CUDA)' if GPU_AVAILABLE else 'CPU'}")
    
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if GPU_AVAILABLE:
        """GPU path but load in 4-bit which is like cuts the 
        memory than the original fine tune, much more faster"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, 
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        #CPU Path
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, 
            torch_dtype = torch.float_32,
            low_cpu_mem_usage=True,
        )
    if IPEX_AVAILABLE:
        print("intel_extension_for_pytorch found -- will apply IPEX optimizations.")
    else:
        print(
                "intel_extension_for_pytorch NOT found. Training will still work, "
                "but install it for a real speedup on Intel CPUs: "
                "pip install intel-extension-for-pytorch --break-system-packages"
            )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="None",
        task_type="CAUSAL_LM",
        target_module=["q_proj","k_proj","v_proj","o_proj"]
        
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer
        
#-------4. Timing check----
def run_timing_check(model, tokenizer, dataset, model_key: str):
    print("\n=== TIMING CHECK ===")
    small_ds = dataset.select(range(min(8, len(dataset))))
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    agrs = TrainingArguments(
        output_dirs="/tmp/braingpt_timing_check",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=5,
        logging_steps=1,
        save_strategy="no",
        use_cpu=not GPU_AVAILABLE,
        fp16=GPU_AVAILABLE,
    )    
    trainer = Trainer(model=model, agrs=args, train_dataset=small_ds, data_collator=collator)
    
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    per_step = elapsed / 5
    total_step_per_run = len(dataset)
    est_seconds = per_step * total_step_per_run
    est_hours = est_seconds / 3600
    
    print(f"\n--- Result: ~{per_step:.1f} sec/step on this machine ---")
    print(f"Full dataset has {len(dataset)} examples.")
    print(f"Estimated time for 1 epoch at batch size 1: ~{est_hours:.1f} hours")
    print(f"Estimated time for 3 epochs: ~{est_hours * 3:.1f} hours")
    print(
        "\nIf that's too long: reduce max_length, increase batch size if RAM allows, "
        "check for a GPU partition on HPC cluster, or consider a smaller base "
        "model."
    )
    
#------5.Main--------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune BrainGPT  (Llama-2 or Mistral) with LoRA")
    parser.add_argument("--model", choices=["llama2", "mistral", "mistralai"], required=True)
    parser.add_argument("--data", required=True, help="Path to training_pairs.jsonl")
    parser.add_argument("--output", default="./braingpt-lora-output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("---time-check",
                        action="store_true",
                        help="Run a 5-step timing check on a small slice of data instead of full training")
    args = parser.parse_args()
    
    print(f"loading training pairs from {args.data}")
    pairs = load_training_pairs(args.data)
    print(f"Loaded {len(pairs)} training examples")
    
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    print("Tokenizingh datasets....")
    dataset = build_dataset(pairs, args.model, tokenizer, max_length=agrs.max_length)
    
    if args.time_check:
        run_timing_check(model, tokenizer, dataset, args.model)
        return
    
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)
    training_args = TrainingArguments(
        output_dirs=args.output,
        per_device_train_batch_Size=args.batch_size,
        num_train_epochs=args.epochs,
         save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        use_cpu=not GPU_AVAILABLE,
        # fp16 mixed precision on GPU (real speed/memory win); fp32 on CPU
        # since mixed precision isn't well supported there.
        fp16=GPU_AVAILABLE,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
 
    print(f"\nStarting training: {args.epochs} epoch(s), batch size {args.batch_size}...")
    trainer.train()
 
    print(f"Saving LoRA adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")
 
 
if __name__ == "__main__":
    main()