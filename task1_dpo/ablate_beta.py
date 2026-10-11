from __future__ import annotations

import argparse
import subprocess
from common.data import load_yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    betas = cfg["betas"]
    short_examples = cfg["short_ablation_examples"]
    print(f"Required beta values: {betas}")
    print(f"Short-run examples per condition: {short_examples}")
    
    for beta in betas:
        run_name = f"dpo_beta_{beta}"
        output_path = f"results/{run_name}"
        print(f"--- Launching ablation fork for beta={beta} ---")
        
        cmd = [
            "python", "-m", "task1_dpo.train",
            "--config", args.config,
            "--run-name", run_name,
            "--beta", str(beta),
            "--max-examples", str(short_examples),
            "--output", output_path
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Finished {run_name}. Saved to {output_path}.\n")

if __name__ == "__main__":
    main()
