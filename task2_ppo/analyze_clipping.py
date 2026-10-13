from __future__ import annotations

import argparse
import torch
import subprocess
from tqdm import tqdm

from common.data import load_yaml, repo_path
from task2_ppo.continue_train import prepare_ppo_continuation, get_logprobs, get_values, safe_to_device, clear_gpu
from task2_ppo.ppo import compute_gae, shaped_rewards, ppo_policy_loss, normalize_advantages

def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print(f"Cached PPO rollouts: {len(rows)}")
    epsilons = cfg["clip_values"]
    print(f"Required epsilon values: {epsilons}")
    
    print("\n--- Evaluating Clipped Surrogate on Cached Batch ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = prepare_ppo_continuation(args.config)
    tokenizer = bundle["tokenizer"]
    policy = safe_to_device(bundle["policy"], device)
    value_model = safe_to_device(bundle["value_model"], device)
    reward_model = safe_to_device(bundle["reward_model"], device)
    
    policy.eval()
    value_model.eval()
    reward_model.eval()
    
    beta = float(cfg.get("kl_beta", 0.1))
    gamma = float(cfg.get("gamma", 1.0))
    lam = float(cfg.get("gae_lambda", 0.95))
    
    with torch.no_grad():
        for row in tqdm(rows[:5], desc="Analyzing sample cache rows"):
            inputs = tokenizer(row["response"], return_tensors="pt", padding=True, truncation=True).to(device)
            attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()
            response_mask = (inputs["input_ids"] != tokenizer.pad_token_id).float()
            
            reward_outputs = reward_model(input_ids=inputs["input_ids"], attention_mask=attention_mask)
            task_rewards = reward_outputs.logits.squeeze(-1).float()
            
            new_logprobs, shift_mask, _ = get_logprobs(policy, inputs["input_ids"], attention_mask, response_mask)
            old_values = get_values(value_model, inputs["input_ids"], attention_mask, response_mask)
            
            old_logprobs = torch.tensor(row["old_logprobs"]).unsqueeze(0).to(device).float()
            ref_logprobs = torch.tensor(row["ref_logprobs"]).unsqueeze(0).to(device).float()
            
            seq_len = shift_mask.shape[1]
            if old_logprobs.shape[1] < seq_len:
                pad_size = seq_len - old_logprobs.shape[1]
                old_logprobs = F.pad(old_logprobs, (0, pad_size), value=0.0)
                ref_logprobs = F.pad(ref_logprobs, (0, pad_size), value=0.0)
            elif old_logprobs.shape[1] > seq_len:
                old_logprobs = old_logprobs[:, :seq_len]
                ref_logprobs = ref_logprobs[:, :seq_len]
            
            step_rewards = shaped_rewards(task_rewards, old_logprobs, ref_logprobs, shift_mask, beta)
            advantages, returns = compute_gae(step_rewards, old_values, shift_mask, gamma=gamma, lam=lam)
            advantages = normalize_advantages(advantages, shift_mask)
            
            for eps in epsilons:
                pol_loss, _, clip_frac = ppo_policy_loss(new_logprobs.float(), old_logprobs.float(), advantages.float(), shift_mask, float(eps))
                print(f"Epsilon: {eps} | Surrogate Loss: {pol_loss.item():.4f} | Clipped Fraction: {clip_frac.item():.4f}")

    clear_gpu()

    print("\n--- Launching Matched Short Forks ---")
    updates = cfg["fork_updates"]
    
    for eps in epsilons:
        run_name = f"ppo_clip_{eps}"
        output_path = f"results/{run_name}"
        print(f"\nStarting fork: {run_name} (Updates: {updates})")
        subprocess.run([
            "python", "-m", "task2_ppo.continue_train",
            "--config", args.config,
            "--run-name", run_name,
            "--clip-epsilon", str(eps),
            "--updates", str(updates),
            "--output", output_path
        ], check=True)

if __name__ == "__main__":
    main()
