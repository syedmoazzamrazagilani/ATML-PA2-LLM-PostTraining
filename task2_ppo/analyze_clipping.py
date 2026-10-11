from __future__ import annotations

import argparse
import torch
import subprocess

from common.data import load_yaml, repo_path

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
    
    print("--- Evaluating Clipped Surrogate on Cached Batch ---")
    
    print("\n--- Launching Matched Short Forks ---")
    updates = cfg["fork_updates"]
    
    for eps in epsilons:
        run_name = f"ppo_clip_{eps}"
        output_path = f"results/{run_name}"
        print(f"Starting fork: {run_name} (Updates: {updates})")
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
