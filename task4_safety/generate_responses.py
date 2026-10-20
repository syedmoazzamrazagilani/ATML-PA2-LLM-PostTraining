from __future__ import annotations

import argparse
import os
import subprocess
import sys
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
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


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


PPO_SAFE_RUNNER_CODE = """
import sys, torch, torch.nn.functional as F
import task2_ppo.continue_train as m

def _mem_safe_get_logprobs(policy, input_ids, attention_mask, response_mask):
    logits = policy(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    with torch.no_grad():
        entropy = torch.zeros(shifted_logits.shape[:2], device=logits.device, dtype=torch.float32)
        for i in range(shifted_logits.shape[1]):
            step_logits = shifted_logits[:, i, :].float()
            step_probs = F.softmax(step_logits, dim=-1)
            step_logps = F.log_softmax(step_logits, dim=-1)
            entropy[:, i] = -torch.sum(step_probs * step_logps, dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0)
    logp = F.log_softmax(shifted_logits.float(), dim=-1)
    del logits, shifted_logits
    logprobs = torch.gather(logp, -1, shifted_labels.unsqueeze(-1)).squeeze(-1)
    del logp
    logprobs = torch.nan_to_num(logprobs, nan=0.0)
    shifted_response_mask = response_mask[:, 1:].float()
    logprobs = torch.where(shifted_response_mask.bool(), logprobs, 0.0)
    entropy = torch.where(shifted_response_mask.bool(), entropy, 0.0)
    return logprobs, shifted_response_mask, entropy

m.get_logprobs = _mem_safe_get_logprobs
sys.argv = ['task2_ppo.continue_train', '--config', 'configs/ppo.yaml', '--run-name', 'standard', '--output', sys.argv[1]]
m.main()
"""


def _ensure_standard_adapter_exists(policy_name: str, adapter: str | None) -> None:
    """If a required Task 1-3 standard adapter is missing in a fresh container, train it first.

    Called BEFORE the parent process initializes CUDA so 100% of GPU memory is available
    to the child training process.
    """
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
            "-c",
            PPO_SAFE_RUNNER_CODE,
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
            f"[Task 4] Standard adapter {adapter} not found on disk; running standard {policy_name.upper()} training in isolated GPU process...",
            flush=True,
        )
        subprocess.run(train_cmds[policy_name], check=True, env=_subprocess_env())


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 8):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]

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

    del model, tokenizer
    clear_gpu()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--policies", nargs="*", default=["sft", "dpo", "ppo", "grpo"])
    ap.add_argument("--single-policy", help="Internal flag to run one policy in an isolated GPU process")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_xstest(cfg)
    expected_rows = len(df)

    if args.single_policy:
        policy_name = args.single_policy
        out_file = outdir / f"generated_{policy_name}.jsonl"
        records = generate_for_policy(cfg, policy_name, batch_size=args.batch_size)
        write_jsonl(out_file, records)
        mean_len = sum(r["response_tokens"] for r in records) / max(1, len(records))
        print(
            f"Saved {len(records)} responses to {out_file} (mean length: {mean_len:.1f} tokens)",
            flush=True,
        )
        return

    print("Policies:", list(args.policies))
    print("XSTest rows:", expected_rows)

    specs = policy_specs(cfg)

    for policy_name in args.policies:
        out_file = outdir / f"generated_{policy_name}.jsonl"
        if not args.overwrite and out_file.exists():
            existing = read_jsonl(out_file)
            if len(existing) == expected_rows:
                continue
        _ensure_standard_adapter_exists(policy_name, specs[policy_name])

    all_records = []
    for policy_name in args.policies:
        out_file = outdir / f"generated_{policy_name}.jsonl"
        if not args.overwrite and out_file.exists():
            existing = read_jsonl(out_file)
            if len(existing) == expected_rows:
                mean_len = sum(r["response_tokens"] for r in existing) / max(1, len(existing))
                print(
                    f"[Task 4] Reusing existing {out_file} ({len(existing)} rows, mean length: {mean_len:.1f} tokens)",
                    flush=True,
                )
                all_records.extend(existing)
                continue

        print(f"\n--- Generating XSTest responses for policy: {policy_name} ---", flush=True)
        cmd = [
            sys.executable,
            "-m",
            "task4_safety.generate_responses",
            "--config",
            args.config,
            "--batch-size",
            str(args.batch_size),
            "--single-policy",
            policy_name,
        ]
        subprocess.run(cmd, check=True, env=_subprocess_env())
        records = read_jsonl(out_file)
        all_records.extend(records)

    write_jsonl(outdir / "generated_all.jsonl", all_records)
    print(f"Completed Task 4 generation for {len(args.policies)} policies.")


if __name__ == "__main__":
    main()
