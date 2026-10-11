from __future__ import annotations

import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task1_dpo.dpo import dpo_loss

def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            try:
                chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
                rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
            except ValueError:
                continue
                
        if not chosen:
            return None, None
            
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate

def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }

def get_response_logps(model, batch, device):
    """Computes sequence log probabilities over response tokens only."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    
    if "labels" in batch:
        mask = (batch["labels"][:, 1:] != -100).to(logits.dtype)
    else:
        mask = attention_mask[:, 1:].to(logits.dtype)
        
    per_token_logps = torch.gather(
        F.log_softmax(shifted_logits, dim=-1), 
        dim=2, 
        index=shifted_labels.unsqueeze(2)
    ).squeeze(-1)
    
    return (per_token_logps * mask).sum(-1)

def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg, model, optimizer, loader = bundle["cfg"], bundle["model"], bundle["optimizer"], bundle["loader"]
    beta_val = bundle["beta"]
    
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    ref_model = load_policy(cfg, trainable=False)
    ref_model.to(device)
    ref_model.eval()

    model.train()
    
    print(f"Starting DPO Training: {run_name} | Beta: {beta_val} | Device: {device}")
    
    for epoch in range(int(cfg.get("epochs", 1))):
        for batch_data in tqdm(loader, desc=f"Epoch {epoch+1}"):
            if batch_data[0] is None:
                continue
                
            chosen_batch, rejected_batch = batch_data
            
            with torch.no_grad():
                ref_chosen_logps = get_response_logps(ref_model, chosen_batch, device)
                ref_rej_logps = get_response_logps(ref_model, rejected_batch, device)
                
            policy_chosen_logps = get_response_logps(model, chosen_batch, device)
            policy_rej_logps = get_response_logps(model, rejected_batch, device)
            
            loss, metrics = dpo_loss(
                policy_chosen_logps, 
                policy_rej_logps, 
                ref_chosen_logps, 
                ref_rej_logps, 
                beta_val
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    print(f"Saving adapter to {output}")
    model.save_pretrained(output)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)

if __name__ == "__main__":
    main()
