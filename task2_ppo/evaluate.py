from __future__ import annotations

import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer

def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    
    bundle = load_evaluation_bundle(args.config, args.adapter)
    tokenizer, model, cfg = bundle["tokenizer"], bundle["policy"], bundle["cfg"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"Evaluating {args.adapter} on {len(bundle['rows'])} rows...")
    
    generated_data = []
    for row in tqdm(bundle["rows"]):
        inputs = tokenizer(row["prompt"], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(cfg.get("max_sequence_length", 256)),
                pad_token_id=tokenizer.eos_token_id
            )
            
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_data.append({
            "prompt": row["prompt"],
            "response": tokenizer.decode(gen_tokens, skip_special_tokens=True),
            "length": len(gen_tokens)
        })
        
    out_path = Path("results") / f"{args.name}_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(generated_data, f, indent=2)

if __name__ == "__main__":
    main()
