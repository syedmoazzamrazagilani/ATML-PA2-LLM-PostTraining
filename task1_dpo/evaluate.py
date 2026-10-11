from __future__ import annotations

import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F

from common.data import load_yaml, read_jsonl, prompt_messages_from_preference, preference_responses, encode_prompt_response, pad_batch
from common.models import load_policy, load_reward_model, load_tokenizer

def load_evaluation_bundle(config_path: str, adapter: str, dataset_path: str | None = None):
    cfg = load_yaml(config_path)
    eval_path = dataset_path or cfg["paths"]["dpo_standard_eval"]
    return {
        "cfg": cfg,
        "rows": read_jsonl(eval_path),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    
    bundle = load_evaluation_bundle(args.config, args.adapter, args.dataset)
    tokenizer = bundle["tokenizer"]
    model = bundle["policy"]
    cfg = bundle["cfg"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"Running evaluation for adapter: {args.adapter} on {len(bundle['rows'])} rows.")
    
    total_length = 0
    generated_data = []
    
    for row in tqdm(bundle["rows"]):
        prompt = prompt_messages_from_preference(row)
        encoded = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(encoded, max_new_tokens=int(cfg["max_sequence_length"]))
            
        generated_tokens = outputs[0][encoded.shape[1]:]
        total_length += len(generated_tokens)
        
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_data.append({
            "prompt": prompt,
            "response": response_text,
            "length": len(generated_tokens)
        })
        
    avg_length = total_length / len(bundle["rows"])
    print(f"Evaluation complete. Mean response length: {avg_length:.2f} tokens")
    
    out_path = Path("results") / f"{args.name}_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(generated_data, f, indent=2)

if __name__ == "__main__":
    main()
