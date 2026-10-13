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
    clear_gpu,
    reference_mode,
)
from task2_ppo.ppo import compute_gae, shaped_rewards, ppo_policy_loss, value_mse_loss, normalize_advantages

def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 42)))

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
        lr=float(cfg.get("policy_learning_rate", 1e-5)),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg.get("value_lora_learning_rate", 1e-5)),
            head_lr=float(cfg.get("value_head_learning_rate", 5e-5)),
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

def scrub_gradients(model):
    for p in trainable_parameters(model):
        if p.grad is not None:
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

def get_logprobs(policy, input_ids, attention_mask, response_mask):
    logits = policy(input_ids=input_ids, attention_mask=attention_mask).logits
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    
    logprobs = -F.cross_entropy(shifted_logits.to(torch.float32).transpose(1, 2), shifted_labels, reduction="none")
    logprobs = torch.nan_to_num(logprobs, nan=0.0)
    
    with torch.no_grad():
        entropy = torch.zeros(shifted_logits.shape[:2], device=logits.device, dtype=torch.float32)
        for i in range(shifted_logits.shape[1]):
            step_logits = shifted_logits[:, i, :].to(torch.float32)
            step_probs = F.softmax(step_logits, dim=-1)
            step_logps = F.log_softmax(step_logits, dim=-1)
            entropy[:, i] = -torch.sum(step_probs * step_logps, dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0)
    
    shifted_response_mask = response_mask[:, 1:].float()
    
    logprobs = torch.where(shifted_response_mask.bool(), logprobs, 0.0)
    entropy = torch.where(shifted_response_mask.bool(), entropy, 0.0)
    
    return logprobs, shifted_response_mask, entropy

def get_values(value_model, input_ids, attention_mask, response_mask):
    values = token_values(value_model, input_ids, attention_mask).float()
    values = torch.nan_to_num(values, nan=0.0) 
    
    shifted_values = values[:, :-1]
    shifted_response_mask = response_mask[:, 1:].float()
    return shifted_values * shifted_response_mask

def safe_to_device(model, device):
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
    
    out = repo_path(output or cfg.get("output", "results/ppo_standard"))
    out.parent.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    policy = safe_to_device(bundle["policy"], device)
    value_model = safe_to_device(bundle["value_model"], device)
    reward_model = safe_to_device(bundle["reward_model"], device)
    tokenizer = bundle["tokenizer"]
    pol_opt, val_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    
    reward_model.eval()

    print(f"--- Starting PPO: {run_name} | Updates: {total_updates} | Eps: {eps} | Beta: {beta} ---")
    torch.cuda.reset_peak_memory_stats()
    
    batch_size = int(cfg.get("prompts_per_update", 1))
    loader = DataLoader(bundle["prompt_rows"], batch_size=batch_size, shuffle=True)
    loader_iter = iter(loader)

    for update in tqdm(range(total_updates), desc="PPO Updates"):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
            
        prompts = [row["prompt"] if isinstance(row, dict) else row for row in batch]
        max_prompt_len = int(cfg.get("max_prompt_length", 256))
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        
        policy.eval()
        with torch.no_grad():
            outputs = policy.generate(
                **inputs,
                max_new_tokens=int(cfg.get("max_response_length", 512)),
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
            
        attention_mask = (outputs != tokenizer.pad_token_id).long()
        response_mask = (outputs != tokenizer.pad_token_id).float()
        prompt_lengths = inputs["input_ids"].shape[1]
        response_mask[:, :prompt_lengths] = 0.0

        with torch.no_grad():
            reward_outputs = reward_model(input_ids=outputs, attention_mask=attention_mask)
            task_rewards = reward_outputs.logits.squeeze(-1).float()
            task_rewards = torch.nan_to_num(task_rewards, nan=0.0)
            
            has_eos = (outputs == tokenizer.eos_token_id).any(dim=-1)
            task_rewards = torch.where(has_eos, task_rewards, task_rewards - float(cfg.get("missing_eos_penalty", 1.0)))
            
            del reward_outputs
            clear_gpu()

            with reference_mode(policy):
                ref_logprobs, _, _ = get_logprobs(policy, outputs, attention_mask, response_mask)
            clear_gpu()
            
            old_logprobs, shift_mask, _ = get_logprobs(policy, outputs, attention_mask, response_mask)
            old_values = get_values(value_model, outputs, attention_mask, response_mask)
            
            step_rewards = shaped_rewards(task_rewards, old_logprobs, ref_logprobs, shift_mask, beta)
            advantages, returns = compute_gae(step_rewards, old_values, shift_mask, gamma=float(cfg.get("gamma", 1.0)), lam=float(cfg.get("gae_lambda", 0.95)))
            
            advantages = torch.nan_to_num(normalize_advantages(advantages, shift_mask), nan=0.0)
            returns = torch.nan_to_num(returns, nan=0.0)

        del ref_logprobs, step_rewards, old_values
        clear_gpu()

        policy.train()
        value_model.train()
        
        ppo_epochs = int(cfg.get("ppo_epochs", 2))
        max_grad = float(cfg.get("max_grad_norm", 1.0))
        
        pol_loss_val = 0.0
        val_loss_val = 0.0
        clip_frac_val = 0.0
        
        for epoch_idx in range(ppo_epochs):
            pol_opt.zero_grad()
            val_opt.zero_grad()

            new_logprobs, _, entropy_mask = get_logprobs(policy, outputs, attention_mask, response_mask)
            new_values = get_values(value_model, outputs, attention_mask, response_mask)
            
            pol_loss, _, clip_frac = ppo_policy_loss(new_logprobs.float(), old_logprobs.detach().float(), advantages.detach().float(), shift_mask, eps)
            val_loss = value_mse_loss(new_values.float(), returns.detach().float(), shift_mask)
            
            loss = pol_loss + float(cfg.get("value_coef", 0.5)) * val_loss
            loss.backward()
            
            scrub_gradients(policy)
            scrub_gradients(value_model)
            
            torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad)
            torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad)
            
            pol_opt.step()
            val_opt.step()
            
            pol_loss_val = pol_loss.item()
            val_loss_val = val_loss.item()
            clip_frac_val = clip_frac.item()
            
            del new_logprobs, new_values, loss, pol_loss, val_loss
            clear_gpu()

        mean_entropy = (entropy_mask.sum() / shift_mask.sum()).item()

        if update % 5 == 0 or update == total_updates - 1:
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            elapsed = time.time() - start_time
            avg_len = shift_mask.sum(dim=1).mean().item()
            print(f"Up {update} | Rew: {task_rewards.mean().item():.2f} | PolLoss: {pol_loss_val:.2f} | ValLoss: {val_loss_val:.2f} | Clip: {clip_frac_val:.3f} | Ent: {mean_entropy:.2f} | Len: {avg_len:.0f} | VRAM: {peak_vram:.2f}GB | {elapsed:.0f}s")

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
