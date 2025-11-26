import argparse
import json
import os
import random
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # Used for loading LoRA adapters

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def main(model_name, output_filename, lora_path=None):
    """
    Runs the MATH-500 test set evaluation.
    """
    # 1. Load MATH-500 test set
    print("Loading MATH-500 test set...")
    ds = load_dataset("ricdomolm/MATH-500", split="test")
    prompts = ds["problem"]   # Question text
    # MATH-500 dataset usually uses 'solution' column for the answer
    gold_answers = ds["solution"] if "solution" in ds.column_names else ds["answer"]

    # 2. Initialize Tokenizer
    # 如果是本地微调路径且缺少tokenizer文件，这里会报错。
    # 为了保险，我们强制使用原始基座模型来加载 Tokenizer（因为微调通常不改词表）
    base_model_name = "Qwen/Qwen3-0.6B-Base"  # <--- 确保这里是你用的基座模型名字
    
    print(f"Loading tokenizer from base: {base_model_name}")
    try:
        # 尝试从 model_name 加载（以防万一你确实保存了）
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    except Exception:
        # 如果失败（比如报错 repo id），则从基座加载
        print(f"Warning: Could not load tokenizer from {model_name}. Fallback to {base_model_name}.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, padding_side='left')
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =================================================================
    # CRITICAL: MATCHING PREPARE.PY LOGIC
    # =================================================================
    
    # 1. The exact instruction string from prepare.py line 78
    instruction = "\nPlease reason step by step, and put your final answer within \\boxed{}."

    print("Preparing prompts with instruction...")
    # 2. Construct Chat Structure
    prompt_chats = [
        [
            {"role": "user", "content": p + instruction}
        ]
        for p in prompts
    ]

    # 3. Apply Chat Template
    # consistent with prepare.py: enable_thinking=False
    # different from prepare.py: add_generation_prompt=True (because we need the model to reply now)
    prompt_strs = [
        tokenizer.apply_chat_template(
            conversation=chat,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        for chat in prompt_chats
    ]
    # =================================================================

    # 3. Load Model
    print(f"Loading base model: {model_name}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto"
    )

    # 4. Load LoRA Adapter (if provided)
    if lora_path:
        print(f"LoRA adapter specified. Loading from local path: {lora_path}")
        if not os.path.isdir(lora_path):
            raise ValueError(f"LoRA path '{lora_path}' is not a valid directory.")
            
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload() # Merge weights for faster inference
        print("Successfully merged LoRA adapter into the base model.")
    else:
        print("No LoRA adapter specified, running the base model.")
    
    model.eval()

    # Generation parameters
    generation_kwargs = {
        "temperature": 0.7,     # Slightly lower temp often helps math reasoning
        "top_p": 0.95,
        "max_new_tokens": 512, # Math problems might need long reasoning
        "do_sample": True,
    }

    # 5. Generate in batches
    batch_size = 32 # Adjust based on your GPU memory (8 or 16 is usually safe for 0.6B model)
    results = []
    
    print(f"Starting generation for {len(prompt_strs)} examples...")
    
    for i in tqdm(range(0, len(prompt_strs), batch_size), desc="Generating"):
        batch_prompts = prompt_strs[i : i + batch_size]
        
        # Tokenize
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        
        # Decode (skip the input prompt part)
        generated_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Store results
        for idx, gen_text in enumerate(generated_texts):
            orig_idx = i + idx
            results.append({
                "id": orig_idx,
                "prompt": prompts[orig_idx],  # Original raw prompt
                "answer": gen_text,           # Model generation
                "gold": gold_answers[orig_idx] # Gold standard
            })

    # 6. Save to JSONL
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"Saved {len(results)} generations to {output_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MATH-500 evaluation.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B-Base", help="Base model name.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output JSONL.")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA checkpoint folder (optional).")
    
    args = parser.parse_args()
    main(args.model, args.output_file, args.lora_path)