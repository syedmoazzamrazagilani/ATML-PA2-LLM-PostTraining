from __future__ import annotations

import argparse
import subprocess
import sys
import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def _ensure_standard_adapter_exists(policy_name: str, adapter: str | None) -> None:
    """If a required Task 1-3 standard adapter is missing in a fresh container, train it first."""
    if adapter is None:
        return
    adapter_path = repo_path(adapter)
    if adapter_path.exists() and any(adapter_path.iterdir()):
        return

    train_cmds = {
        "dpo": [
            sys.executable,
            "-m",
            "task1_dpo.train",
            "--config",
            "configs/dpo.yaml",
            "--run-name",
            "standard",
            "--output",
            str(adapter),
        ],
        "ppo": [
            sys.executable,
            "-m",
            "task2_ppo.continue_train",
            "--config",
            "configs/ppo.yaml",
            "--run-name",
            "standard",
            "--output",
            str(adapter),
        ],
        "grpo": [
            sys.executable,
            "-m",
            "task3_grpo.continue_train",
            "--config",
            "configs/grpo.yaml",
            "--run-name",
            "standard",
            "--output",
            str(adapter),
        ],
    }
    if policy_name in train_cmds:
        print(
            f"[Task 4] Standard adapter {adapter} not found on disk; running standard {policy_name.upper()} training first..."
        )
        subprocess.run(train_cmds[policy_name], check=True)


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 8):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    _ensure_standard_adapter_exists(policy_name, adapter)

    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    model.eval()
    model.config.use_cache = True

    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok, trunc in zip(
            chunk.iterrows(), gen["responses"], gen["response_lengths"], gen["truncated"]
        ):
            records.append(
                {
                    "xstest_id": int(row["xstest_id"]),
                    "policy": policy_name,
                    "prompt": str(row["prompt"]),
                    "benchmark_class": str(row["benchmark_class"]).upper(),
                    "type": str(row["type"]),
                    "response": response,
                    "response_tokens": int(n_tok),
                    "truncated": bool(trunc),
                }
            )

    clear_gpu(model, tokenizer)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--policies", nargs="*", default=["sft", "dpo", "ppo", "grpo"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_xstest(cfg)
    print("Policies:", list(args.policies))
    print("XSTest rows:", len(df))

    all_records = []
    for policy_name in args.policies:
        print(f"\n--- Generating XSTest responses for policy: {policy_name} ---")
        records = generate_for_policy(cfg, policy_name, batch_size=args.batch_size)
        out_file = outdir / f"generated_{policy_name}.jsonl"
        write_jsonl(out_file, records)
        all_records.extend(records)
        mean_len = sum(r["response_tokens"] for r in records) / max(1, len(records))
        print(
            f"Saved {len(records)} responses to {out_file} (mean length: {mean_len:.1f} tokens)"
        )

    write_jsonl(outdir / "generated_all.jsonl", all_records)
    print(f"Completed Task 4 generation for {len(args.policies)} policies.")


if __name__ == "__main__":
    main()
