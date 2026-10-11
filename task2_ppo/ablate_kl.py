from __future__ import annotations

import argparse
import subprocess
from common.data import load_yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    kl_betas = cfg["kl_values"]
    updates = cfg["fork_updates"]
    print(f"KL beta conditions: {kl_betas}")
    print(f"Fork update budget: {updates}")
    
    for beta in kl_betas:
        run_name = f"ppo_kl_{beta}"
        output_path = f"results/{run_name}"
        print(f"--- Launching ablation fork for KL beta={beta} ---")
        
        subprocess.run([
            "python", "-m", "task2_ppo.continue_train",
            "--config", args.config,
            "--run-name", run_name,
            "--kl-beta", str(beta),
            "--updates", str(updates),
            "--output", output_path
        ], check=True)
        
        subprocess.run([
            "python", "-m", "task2_ppo.evaluate",
            "--config", args.config,
            "--adapter", output_path,
            "--name", run_name
        ], check=True)

if __name__ == "__main__":
    main()
