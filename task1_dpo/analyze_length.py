from __future__ import annotations

import argparse
import subprocess
from common.data import load_yaml, read_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    balanced_path = cfg["paths"]["dpo_length_train"]
    stratified_path = cfg["paths"]["dpo_length_eval"]
    
    balanced = read_jsonl(balanced_path)
    stratified = read_jsonl(stratified_path)
    print(f"Length-balanced train rows: {len(balanced)}")
    print(f"Length-stratified eval rows: {len(stratified)}")
    
    output_path = "results/dpo_length_balanced"
    
    print("--- Training Length-Balanced DPO ---")
    subprocess.run([
        "python", "-m", "task1_dpo.train",
        "--config", args.config,
        "--run-name", "length_balanced",
        "--dataset", balanced_path,
        "--output", output_path
    ], check=True)
    
    print("--- Evaluating Standard DPO on Length-Stratified Set ---")
    subprocess.run([
        "python", "-m", "task1_dpo.evaluate",
        "--config", args.config,
        "--adapter", cfg["standard_output"],
        "--dataset", stratified_path,
        "--name", "eval_standard_length_stratified"
    ], check=True)
    
    print("--- Evaluating Balanced DPO on Length-Stratified Set ---")
    subprocess.run([
        "python", "-m", "task1_dpo.evaluate",
        "--config", args.config,
        "--adapter", output_path,
        "--dataset", stratified_path,
        "--name", "eval_balanced_length_stratified"
    ], check=True)

if __name__ == "__main__":
    main()
