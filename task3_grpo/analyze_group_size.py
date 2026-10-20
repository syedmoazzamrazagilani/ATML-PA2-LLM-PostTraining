from __future__ import annotations

import argparse
from collections import defaultdict
import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def _sorted_prompt_ids(by_prompt) -> list[str]:
    return sorted(by_prompt.keys(), key=lambda x: (0, int(x)) if str(x).lstrip("-").isdigit() else (1, str(x)))


def regroup_equal_generation_budget(by_prompt, k: int) -> list[dict]:
    """Return K-sized within-prompt groups while keeping total cached completions fixed at 8*N.

    Each prompt has 8 cached completions sorted by `generation_index` (0..7).
    For k in {2, 4, 8}, we partition each prompt's 8 completions into (8 // k) disjoint
    within-prompt groups of size k:
      - K=2: 4 groups/prompt * N prompts = 4N groups (8N total completions)
      - K=4: 2 groups/prompt * N prompts = 2N groups (8N total completions)
      - K=8: 1 group/prompt  * N prompts =  N groups (8N total completions)
    Every group consists solely of completions generated for the same prompt, and all
    conditions use the exact same fixed pool of 8N cached completions.
    """
    k = int(k)
    if k not in {2, 4, 8} or 8 % k != 0:
        raise ValueError(f"Group size k={k} must divide 8 (expected one of {{2, 4, 8}})")

    groups = []
    for pid in _sorted_prompt_ids(by_prompt):
        completions = by_prompt[pid][:8]
        for sub_idx, start in enumerate(range(0, 8, k)):
            chunk = completions[start : start + k]
            groups.append(
                {
                    "group_id": f"{pid}_k{k}_sub{sub_idx}",
                    "source_index": pid,
                    "sub_group_index": sub_idx,
                    "k": k,
                    "rows": chunk,
                    "rewards": [float(r["reward"]) for r in chunk],
                }
            )
    return groups


def regroup_prompt_subset_equal_budget(by_prompt, k: int, prompt_bin_map: dict[str, str]) -> list[dict]:
    """Alternative equal-generation partitioning (budget = 2*N completions) using 1 group/prompt.

    Selects floor(8 / k) * (N / 4) prompts (stratified across difficulty ordering) and takes
    the first K completions (generation_index 0..K-1) per selected prompt so total generations
    equal 2*N for K=2 (N prompts), K=4 (N/2 prompts), and K=8 (N/4 prompts).
    """
    k = int(k)
    pids = _sorted_prompt_ids(by_prompt)
    pids_sorted_by_reward = sorted(
        pids,
        key=lambda p: (float(np.mean([r["reward"] for r in by_prompt[p][:8]])), str(p)),
    )
    stride = k // 2  # K=2 -> stride 1 (all N), K=4 -> stride 2 (N/2), K=8 -> stride 4 (N/4)
    selected_pids = pids_sorted_by_reward[::stride]

    groups = []
    for pid in selected_pids:
        chunk = by_prompt[pid][:k]
        groups.append(
            {
                "group_id": f"{pid}_subset_k{k}",
                "source_index": pid,
                "sub_group_index": 0,
                "k": k,
                "difficulty_bin": prompt_bin_map.get(pid, "unknown"),
                "rows": chunk,
                "rewards": [float(r["reward"]) for r in chunk],
            }
        )
    return groups


def assign_prompt_difficulty_bins(by_prompt) -> tuple[dict[str, dict], dict]:
    """Define prompt difficulty ONCE using each prompt's full K=8 mean reward.

    Provides both:
      - 3-bin terciles (`hard`, `medium`, `easy`: bottom/middle/top third of prompt mean reward)
      - 2-bin median split (`hard_half` <= median, `easy_half` > median)
    """
    pids = _sorted_prompt_ids(by_prompt)
    prompt_means = {}
    prompt_stds = {}
    for pid in pids:
        rews = np.asarray([float(r["reward"]) for r in by_prompt[pid][:8]], dtype=float)
        prompt_means[pid] = float(np.mean(rews))
        prompt_stds[pid] = float(np.std(rews))

    vals = np.asarray([prompt_means[pid] for pid in pids], dtype=float)
    q33, q67 = np.percentile(vals, [33.3333, 66.6667])
    median_val = float(np.median(vals))

    prompt_info = {}
    for pid in pids:
        m = prompt_means[pid]
        if m <= q33:
            tercile_bin = "hard_low_reward"
        elif m <= q67:
            tercile_bin = "medium_mid_reward"
        else:
            tercile_bin = "easy_high_reward"

        median_bin = "hard_below_median" if m <= median_val else "easy_above_median"
        prompt_info[pid] = {
            "prompt_mean_reward_k8": m,
            "prompt_std_reward_k8": prompt_stds[pid],
            "tercile_bin": tercile_bin,
            "median_bin": median_bin,
        }

    bin_rule_meta = {
        "definition": (
            "Prompt difficulty is defined once per prompt by its mean reward across all 8 cached "
            "completions: bar_r_p = (1/8) sum_{j=1}^8 r_{p,j}. Tercile bins partition prompts into "
            "hard_low_reward (<= P33.3), medium_mid_reward (P33.3..P66.7), and easy_high_reward (> P66.7); "
            "median bins split prompts at the median (<= P50 vs > P50)."
        ),
        "tercile_thresholds": {"q33": float(q33), "q67": float(q67)},
        "median_threshold": median_val,
        "overall_prompt_mean_reward": float(np.mean(vals)),
        "overall_prompt_std_reward": float(np.std(vals)),
    }
    return prompt_info, bin_rule_meta


def compute_metrics_for_groups(groups: list[dict], prompt_info: dict[str, dict], eps: float = 1e-6) -> dict:
    if not groups:
        return {}

    flat_rewards = []
    flat_group_ids = []
    group_stds = []
    group_vars = []
    group_ranges = []
    group_means = []
    informative_flags = []

    by_pid_stds = defaultdict(list)

    for g_idx, g in enumerate(groups):
        rews = np.asarray(g["rewards"], dtype=float)
        std_val = float(np.std(rews, ddof=0))
        var_val = float(np.var(rews, ddof=0))
        rng_val = float(np.max(rews) - np.min(rews))
        mean_val = float(np.mean(rews))
        is_inf = bool(std_val > eps)

        group_stds.append(std_val)
        group_vars.append(var_val)
        group_ranges.append(rng_val)
        group_means.append(mean_val)
        informative_flags.append(1.0 if is_inf else 0.0)
        by_pid_stds[g["source_index"]].append(std_val)

        for r in rews:
            flat_rewards.append(float(r))
            flat_group_ids.append(g_idx)

    rewards_t = torch.tensor(flat_rewards, dtype=torch.float32)
    gids_t = torch.tensor(flat_group_ids, dtype=torch.long)
    adv_t = group_relative_advantages(rewards_t, gids_t, eps=eps).numpy()

    std_estimation_errors = []
    for pid, sub_stds in by_pid_stds.items():
        ref_std = prompt_info[pid]["prompt_std_reward_k8"]
        for s in sub_stds:
            std_estimation_errors.append((s - ref_std) ** 2)

    group_stds_arr = np.asarray(group_stds, dtype=float)
    group_vars_arr = np.asarray(group_vars, dtype=float)
    inf_arr = np.asarray(informative_flags, dtype=float)

    return {
        "num_groups": int(len(groups)),
        "total_generations": int(len(flat_rewards)),
        "informative_group_rate": float(np.mean(inf_arr)),
        "uninformative_group_rate": float(1.0 - np.mean(inf_arr)),
        "mean_within_group_reward_std": float(np.mean(group_stds_arr)),
        "median_within_group_reward_std": float(np.median(group_stds_arr)),
        "std_of_within_group_reward_std": float(np.std(group_stds_arr)),
        "variance_of_within_group_reward_std": float(np.var(group_stds_arr)),
        "mean_within_group_reward_variance": float(np.mean(group_vars_arr)),
        "group_relative_advantage_variance": float(np.var(adv_t)),
        "group_std_rmse_vs_k8": float(np.sqrt(np.mean(std_estimation_errors))) if std_estimation_errors else 0.0,
        "mean_within_group_reward_range": float(np.mean(group_ranges)),
        "mean_reward": float(np.mean(flat_rewards)),
    }


def run_group_size_study(config_path: str = "configs/grpo.yaml") -> dict:
    cfg = load_yaml(config_path)
    by_prompt = load_k8_cache(cfg["group_cache"])
    group_sizes = [int(k) for k in cfg.get("group_sizes", [2, 4, 8])]

    prompt_info, bin_rule_meta = assign_prompt_difficulty_bins(by_prompt)
    prompt_tercile_map = {pid: info["tercile_bin"] for pid, info in prompt_info.items()}

    results_by_k = {}
    subset_results_by_k = {}

    for k in group_sizes:
        groups = regroup_equal_generation_budget(by_prompt, k)
        overall = compute_metrics_for_groups(groups, prompt_info)

        by_tercile = {}
        for bin_name in ["hard_low_reward", "medium_mid_reward", "easy_high_reward"]:
            bin_groups = [g for g in groups if prompt_info[g["source_index"]]["tercile_bin"] == bin_name]
            by_tercile[bin_name] = compute_metrics_for_groups(bin_groups, prompt_info)

        by_median = {}
        for bin_name in ["hard_below_median", "easy_above_median"]:
            bin_groups = [g for g in groups if prompt_info[g["source_index"]]["median_bin"] == bin_name]
            by_median[bin_name] = compute_metrics_for_groups(bin_groups, prompt_info)

        results_by_k[str(k)] = {
            "k": k,
            "overall": overall,
            "by_tercile_difficulty": by_tercile,
            "by_median_difficulty": by_median,
        }

        sub_groups = regroup_prompt_subset_equal_budget(by_prompt, k, prompt_tercile_map)
        sub_overall = compute_metrics_for_groups(sub_groups, prompt_info)
        sub_by_tercile = {}
        for bin_name in ["hard_low_reward", "medium_mid_reward", "easy_high_reward"]:
            b_groups = [g for g in sub_groups if prompt_info[g["source_index"]]["tercile_bin"] == bin_name]
            sub_by_tercile[bin_name] = compute_metrics_for_groups(b_groups, prompt_info)
        subset_results_by_k[str(k)] = {
            "k": k,
            "overall": sub_overall,
            "by_tercile_difficulty": sub_by_tercile,
        }

    pids_by_std = sorted(
        _sorted_prompt_ids(by_prompt),
        key=lambda p: prompt_info[p]["prompt_std_reward_k8"],
    )
    low_var_pid = pids_by_std[0]
    high_var_pid = pids_by_std[-1]
    qualitative_examples = {
        "low_variation_prompt": {
            "source_index": low_var_pid,
            "difficulty_bin": prompt_info[low_var_pid]["tercile_bin"],
            "mean_reward_k8": prompt_info[low_var_pid]["prompt_mean_reward_k8"],
            "std_reward_k8": prompt_info[low_var_pid]["prompt_std_reward_k8"],
            "rewards_k8": [float(r["reward"]) for r in by_prompt[low_var_pid][:8]],
            "k2_subgroup_stds": [
                float(np.std([float(r["reward"]) for r in by_prompt[low_var_pid][i : i + 2]]))
                for i in range(0, 8, 2)
            ],
            "sample_completion_snippet": str(by_prompt[low_var_pid][0].get("completion", ""))[:300],
        },
        "high_variation_prompt": {
            "source_index": high_var_pid,
            "difficulty_bin": prompt_info[high_var_pid]["tercile_bin"],
            "mean_reward_k8": prompt_info[high_var_pid]["prompt_mean_reward_k8"],
            "std_reward_k8": prompt_info[high_var_pid]["prompt_std_reward_k8"],
            "rewards_k8": [float(r["reward"]) for r in by_prompt[high_var_pid][:8]],
            "k2_subgroup_stds": [
                float(np.std([float(r["reward"]) for r in by_prompt[high_var_pid][i : i + 2]]))
                for i in range(0, 8, 2)
            ],
            "sample_completion_snippet": str(by_prompt[high_var_pid][0].get("completion", ""))[:300],
        },
    }

    output_obj = {
        "num_cached_prompts": int(len(by_prompt)),
        "total_cached_completions": int(len(by_prompt) * 8),
        "group_sizes": group_sizes,
        "binning_rule": bin_rule_meta,
        "equal_generation_within_prompt_regrouping": results_by_k,
        "equal_generation_prompt_subset_regrouping": subset_results_by_k,
        "qualitative_cache_examples": qualitative_examples,
    }

    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    out_path = results_dir / "group_size_study.json"
    save_json(out_path, output_obj)
    save_json(repo_path("results") / "grpo_group_size_study.json", output_obj)

    print(f"Cached prompts: {len(by_prompt)} | Total cached completions: {len(by_prompt) * 8}")
    print("Binning rule:", bin_rule_meta["definition"])
    for k_str, k_res in results_by_k.items():
        ov = k_res["overall"]
        print(
            f"K={k_str} (groups={ov['num_groups']}, gens={ov['total_generations']}) | "
            f"Informative={ov['informative_group_rate']:.4f} | "
            f"Uninformative={ov['uninformative_group_rate']:.4f} | "
            f"MeanGrpStd={ov['mean_within_group_reward_std']:.4f} | "
            f"GrpVarMean={ov['mean_within_group_reward_variance']:.4f} | "
            f"StdRMSEvsK8={ov['group_std_rmse_vs_k8']:.4f}"
        )
        for b_name, b_stats in k_res["by_tercile_difficulty"].items():
            print(
                f"   [{b_name}] groups={b_stats['num_groups']} | "
                f"Informative={b_stats['informative_group_rate']:.4f} | "
                f"MeanGrpStd={b_stats['mean_within_group_reward_std']:.4f} | "
                f"GrpVarMean={b_stats['mean_within_group_reward_variance']:.4f}"
            )
    print(f"Saved group-size study to {out_path}")
    return output_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    run_group_size_study(args.config)


if __name__ == "__main__":
    main()
