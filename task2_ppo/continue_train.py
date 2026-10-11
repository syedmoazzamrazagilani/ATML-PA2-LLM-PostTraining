from __future__ import annotations

import argparse
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data import load_yaml, read_jsonl, repo_path, pad_batch
from common.logging_utils import set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
    token_values,
)
from task2_ppo.ppo import compute_gae, shaped_rewards, ppo_policy_loss, value_mse_loss, normalize_advantages

def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }

def get_logprobs_and_values(policy, value_model, input_ids, attention_mask, response_mask):
    logits = policy(input_ids=input_ids, attention_mask=attention_mask).logits
    
    values = token_values(value_model, input_ids, attention_mask)
    
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    
    logprobs = torch.gather(
        F.log_softmax(shifted_logits, dim=-1), 
        dim=2, 
        index=shifted_labels.unsqueeze(2)
    ).squeeze(-1)
    
    shifted_response_mask = response_mask[:, 1:]
    shifted_values = values[:, :-1]
    
    return logprobs * shifted_response_mask, shifted_values * shifted_response_mask, shifted_response_mask
    
def safe_to_device(model, device):
    """Safely moves models to device, bypassing 8-bit quantized models."""
    try:
        return model.to(device)
    except ValueError:
        return model

def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    start_time = time.time()
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    
    total_updates = updates if updates is not None else int(cfg.get("updates", 20))
    eps = clip_epsilon if clip_epsilon is not None else float(cfg.get("clip_epsilon", 0.2))
    beta = kl_beta if kl_beta is not None else float(cfg.get("kl_beta", 0.1))
    
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    policy = safe_to_device(bundle["policy"], device)
    value_model = safe_to_device(bundle["value_model"], device)
    reward_model = safe_to_device(bundle["reward_model"], device)
    tokenizer = bundle["tokenizer"]
    pol_opt, val_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    
    ref_policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    ref_policy = safe_to_device(ref_policy, device)
    
    ref_policy.eval()
    reward_model.eval()

    print(f"--- Starting PPO: {run_name} | Updates: {total_updates} | Eps: {eps} | Beta: {beta} ---")
    torch.cuda.reset_peak_memory_stats()
    
    loader = DataLoader(bundle["prompt_rows"], batch_size=int(cfg["prompts_per_update"]), shuffle=True)
    loader_iter = iter(loader)

    for update in tqdm(range(total_updates), desc="PPO Updates"):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
            
        prompts = [row["prompt"] if isinstance(row, dict) else row for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        policy.eval()
        with torch.no_grad():
            outputs = policy.generate(
                **inputs,
                max_new_tokens=int(cfg["max_response_length"]),
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
            
        response_mask = (outputs != tokenizer.pad_token_id).float()
        prompt_lengths = inputs["input_ids"].shape[1]
        response_mask[:, :prompt_lengths] = 0.0

        with torch.no_grad():
            reward_outputs = reward_model(input_ids=outputs, attention_mask=(outputs != tokenizer.pad_token_id).long())
            task_rewards = reward_outputs.logits.squeeze(-1)

            ref_logprobs, _, _ = get_logprobs_and_values(ref_policy, value_model, outputs, (outputs != tokenizer.pad_token_id).long(), response_mask)
            old_logprobs, old_values, shift_mask = get_logprobs_and_values(policy, value_model, outputs, (outputs != tokenizer.pad_token_id).long(), response_mask)
            
            step_rewards = shaped_rewards(task_rewards, old_logprobs, ref_logprobs, shift_mask, beta)
            advantages, returns = compute_gae(step_rewards, old_values, shift_mask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
            advantages = normalize_advantages(advantages, shift_mask)

        policy.train()
        value_model.train()
        
        pol_opt.zero_grad()
        val_opt.zero_grad()

        new_logprobs, new_values, _ = get_logprobs_and_values(policy, value_model, outputs, (outputs != tokenizer.pad_token_id).long(), response_mask)
        
        pol_loss, _, clip_frac = ppo_policy_loss(new_logprobs, old_logprobs, advantages, shift_mask, eps)
        val_loss = value_mse_loss(new_values, returns, shift_mask)
        
        loss = pol_loss + float(cfg["value_coef"]) * val_loss
        loss.backward()
        
        pol_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), 1.0)
        
        pol_opt.step()
        val_opt.step()

        if update % 5 == 0 or update == total_updates - 1:
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            elapsed = time.time() - start_time
            avg_len = shift_mask.sum(dim=1).mean().item()
            print(f"Update {update} | Reward: {task_rewards.mean().item():.3f} | Pol Loss: {pol_loss.item():.3f} | Val Loss: {val_loss.item():.3f} | Clip Frac: {clip_frac.item():.3f} | Grad Norm: {pol_grad_norm.item():.3f} | Len: {avg_len:.1f} | VRAM: {peak_vram:.2f}GB | Time: {elapsed:.1f}s")

    policy.save_pretrained(out)
    print(f"Finished PPO Training. Adapter saved to {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)

if __name__ == "__main__":
    main()
