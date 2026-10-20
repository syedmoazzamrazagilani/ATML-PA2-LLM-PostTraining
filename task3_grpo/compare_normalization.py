from __future__ import annotations

import argparse
from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_grpo_policy


def _select_qualitative_comparisons(canon_eval: dict, dr_eval: dict, top_n: int = 5) -> list[dict]:
    canon_by_id = {str(r["source_index"]): r for r in canon_eval.get("examples", [])}
    dr_by_id = {str(r["source_index"]): r for r in dr_eval.get("examples", [])}
    common_ids = [pid for pid in canon_by_id if pid in dr_by_id]

    pairs = []
    for pid in common_ids:
        c = canon_by_id[pid]
        d = dr_by_id[pid]
        pairs.append(
            {
                "source_index": pid,
                "prompt": c["prompt"],
                "canonical_length": int(c["length"]),
                "dr_grpo_length": int(d["length"]),
                "length_diff_dr_minus_canonical": int(d["length"] - c["length"]),
                "canonical_reward": float(c["reward"]),
                "dr_grpo_reward": float(d["reward"]),
                "reward_diff_dr_minus_canonical": float(d["reward"] - c["reward"]),
                "canonical_response_snippet": str(c["response"])[:400],
                "dr_grpo_response_snippet": str(d["response"])[:400],
            }
        )
    pairs.sort(key=lambda x: abs(x["length_diff_dr_minus_canonical"]), reverse=True)
    return pairs[:top_n]


def run_normalization_study(config_path: str = "configs/grpo.yaml") -> dict:
    cfg = load_yaml(config_path)
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running Task 3 Step 3: Length-Normalization Study (fork_updates={fork_updates}) ===")

    canon_out = "outputs/task3_grpo/grpo_norm_canonical"
    dr_out = "outputs/task3_grpo/grpo_norm_dr_grpo"

    canon_train = run_grpo(
        config_path=config_path,
        output=canon_out,
        updates=fork_updates,
        loss_type="grpo",
        run_name="grpo_norm_canonical",
    )
    canon_eval = evaluate_grpo_policy(
        config_path=config_path,
        adapter=canon_out,
        name="grpo_norm_canonical",
    )

    dr_train = run_grpo(
        config_path=config_path,
        output=dr_out,
        updates=fork_updates,
        loss_type="dr_grpo",
        run_name="grpo_norm_dr_grpo",
    )
    dr_eval = evaluate_grpo_policy(
        config_path=config_path,
        adapter=dr_out,
        name="grpo_norm_dr_grpo",
    )

    comparison = {
        "fork_updates": fork_updates,
        "max_completion_length": int(cfg.get("max_completion_length", 512)),
        "canonical_grpo": {
            "loss_type": "grpo",
            "train_summary": {k: v for k, v in canon_train.items() if k != "completion_records"},
            "eval_summary": {k: v for k, v in canon_eval.items() if k != "examples"},
        },
        "dr_grpo": {
            "loss_type": "dr_grpo",
            "train_summary": {k: v for k, v in dr_train.items() if k != "completion_records"},
            "eval_summary": {k: v for k, v in dr_eval.items() if k != "examples"},
        },
        "deltas_dr_minus_canonical": {
            "held_out_reward_diff": float(dr_eval["reward_mean"] - canon_eval["reward_mean"]),
            "held_out_sampled_kl_diff": float(
                dr_eval["sampled_kl_token_mean"] - canon_eval["sampled_kl_token_mean"]
            ),
            "held_out_signed_kl_diff": float(
                dr_eval["signed_sampled_kl_token_mean"] - canon_eval["signed_sampled_kl_token_mean"]
            ),
            "held_out_length_mean_diff": float(
                dr_eval["response_length_mean"] - canon_eval["response_length_mean"]
            ),
            "held_out_length_iqr_diff": float(
                dr_eval["response_length_iqr"] - canon_eval["response_length_iqr"]
            ),
            "train_short_to_long_per_token_grad_ratio_canonical": float(
                canon_train["length_conditioned_gradient_stats"]["short_to_long_per_token_grad_ratio"]
            ),
            "train_short_to_long_per_token_grad_ratio_dr_grpo": float(
                dr_train["length_conditioned_gradient_stats"]["short_to_long_per_token_grad_ratio"]
            ),
            "train_corr_length_per_token_grad_canonical": float(
                canon_train["length_conditioned_gradient_stats"]["corr_length_per_token_grad"]
            ),
            "train_corr_length_per_token_grad_dr_grpo": float(
                dr_train["length_conditioned_gradient_stats"]["corr_length_per_token_grad"]
            ),
        },
        "qualitative_examples": _select_qualitative_comparisons(canon_eval, dr_eval),
    }

    out_path = results_dir / "normalization_comparison.json"
    save_json(out_path, comparison)
    save_json(repo_path("results") / "grpo_normalization_comparison.json", comparison)

    print("\n=== Canonical GRPO vs Dr. GRPO Summary ===")
    print(
        f"Canonical GRPO | Held-out Reward: {canon_eval['reward_mean']:.4f} | "
        f"KL: {canon_eval['sampled_kl_token_mean']:.4f} | "
        f"Len: {canon_eval['response_length_mean']:.1f} ± {canon_eval['response_length_std']:.1f} | "
        f"Short/Long TokGrad Ratio: {canon_train['length_conditioned_gradient_stats']['short_to_long_per_token_grad_ratio']:.3f}"
    )
    print(
        f"Dr. GRPO       | Held-out Reward: {dr_eval['reward_mean']:.4f} | "
        f"KL: {dr_eval['sampled_kl_token_mean']:.4f} | "
        f"Len: {dr_eval['response_length_mean']:.1f} ± {dr_eval['response_length_std']:.1f} | "
        f"Short/Long TokGrad Ratio: {dr_train['length_conditioned_gradient_stats']['short_to_long_per_token_grad_ratio']:.3f}"
    )
    print(f"Saved normalization comparison to {out_path}")
    return comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    run_normalization_study(args.config)


if __name__ == "__main__":
    main()
