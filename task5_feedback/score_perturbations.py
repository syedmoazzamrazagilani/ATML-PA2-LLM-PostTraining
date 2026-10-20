from __future__ import annotations

import argparse
from collections import defaultdict
import re
import numpy as np
import pandas as pd

from common.data import last_assistant_text, load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import clear_gpu
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

ORDERED_VARIANTS = [
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "persuasive_filler_correct",
    "good_reasoning_wrong_final",
    "gold_distractor_wrong_final",
]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        pid = str(row.get("problem_id", row.get("source_index", row.get("id", ""))))
        by_problem[pid][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def _sorted_problem_ids(by_problem: dict) -> list[str]:
    return sorted(
        by_problem.keys(),
        key=lambda x: (0, int(x)) if str(x).lstrip("-").isdigit() else (1, str(x)),
    )


def _extract_problem_text(variants: dict) -> str:
    for v in variants.values():
        for k in ("question", "problem", "prompt"):
            val = v.get(k)
            if isinstance(val, str) and val.strip():
                return val.strip()
        msgs = v.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    return str(m.get("content", "")).strip()
    return ""


def _extract_response_text(row: dict) -> str:
    for k in ("response", "completion", "candidate", "output", "text", "solution"):
        val = row.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    if "messages" in row:
        return last_assistant_text(row["messages"]).strip()
    if isinstance(row.get("answer"), str) and "####" in row["answer"]:
        return row["answer"].strip()
    raise KeyError(f"Could not locate response text in diagnostic row keys={list(row.keys())}")


def _extract_gold_final(variants: dict) -> str:
    for v_name in ["clean_correct"] + ORDERED_VARIANTS:
        row = variants.get(v_name, {})
        for k in ("gold_final", "gold", "target", "final_answer", "ground_truth"):
            val = row.get(k)
            if val is not None and str(val).strip() != "":
                ext = extract_designated_final(str(val))
                return ext if ext is not None else str(val).strip().replace(",", "")
    clean_resp = _extract_response_text(variants["clean_correct"])
    ext = extract_designated_final(clean_resp)
    if ext is not None:
        return ext
    raise KeyError("Could not resolve gold_final for diagnostic problem group")


def _rlvr_pref(rew_a: float, rew_b: float) -> str:
    if rew_a > rew_b:
        return "A"
    if rew_b > rew_a:
        return "B"
    return "TIE"


def _evaluate_pair_list(pair_records: list[dict], pair_name: str, category_group: str) -> dict:
    n = max(1, len(pair_records))

    rlvr_better = sum(1 for r in pair_records if r["rlvr_pref"] == "A")
    rlvr_tie = sum(1 for r in pair_records if r["rlvr_pref"] == "TIE")
    rlvr_wrong = sum(1 for r in pair_records if r["rlvr_pref"] == "B")

    rlaif_better = sum(1 for r in pair_records if r["rlaif_pref"] == "A")
    rlaif_tie = sum(1 for r in pair_records if r["rlaif_pref"] == "TIE")
    rlaif_wrong = sum(1 for r in pair_records if r["rlaif_pref"] == "B")

    agree_count = sum(1 for r in pair_records if r["rlvr_pref"] == r["rlaif_pref"])

    return {
        "pair_name": pair_name,
        "category_group": category_group,
        "better_variant": pair_records[0]["variant_a"] if pair_records else "",
        "perturbed_variant": pair_records[0]["variant_b"] if pair_records else "",
        "num_pairs": int(len(pair_records)),
        "rlvr_better_response_rate": float(rlvr_better / n),
        "rlvr_tie_rate": float(rlvr_tie / n),
        "rlvr_wrong_preference_rate": float(rlvr_wrong / n),
        "rlvr_win_rate": float((rlvr_better + 0.5 * rlvr_tie) / n),
        "rlvr_counts": {"better": int(rlvr_better), "tie": int(rlvr_tie), "wrong": int(rlvr_wrong)},
        "rlaif_better_response_rate": float(rlaif_better / n),
        "rlaif_tie_rate": float(rlaif_tie / n),
        "rlaif_wrong_preference_rate": float(rlaif_wrong / n),
        "rlaif_win_rate": float((rlaif_better + 0.5 * rlaif_tie) / n),
        "rlaif_counts": {"better": int(rlaif_better), "tie": int(rlaif_tie), "wrong": int(rlaif_wrong)},
        "verifier_judge_agreement_rate": float(agree_count / n),
    }


def run_perturbation_study(config_path: str = "configs/feedback.yaml") -> dict:
    timer = wall_timer()
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    pids = _sorted_problem_ids(groups)
    print(f"=== Task 5 Step 2: Controlled Reward Diagnostic Study ({len(pids)} problems, 100 responses) ===")

    cache_path = outdir / "pairwise_judge_cache_perturbations.json"
    judge = PairwiseAIJudge(cfg, cache_path=cache_path)

    variant_rows = []
    pair_specs = [
        (
            "clean_vs_corrupt_reasoning_correct_final",
            "reasoning_sensitivity",
            "clean_correct",
            "corrupt_reasoning_correct_final",
        ),
        (
            "clean_vs_persuasive_filler_correct",
            "style_filler_robustness",
            "clean_correct",
            "persuasive_filler_correct",
        ),
        (
            "clean_vs_good_reasoning_wrong_final",
            "outcome_sensitivity",
            "clean_correct",
            "good_reasoning_wrong_final",
        ),
        (
            "clean_vs_gold_distractor_wrong_final",
            "outcome_distractor_robustness",
            "clean_correct",
            "gold_distractor_wrong_final",
        ),
        (
            "corrupt_reasoning_correct_final_vs_good_reasoning_wrong_final",
            "outcome_vs_reasoning_tradeoff",
            "corrupt_reasoning_correct_final",
            "good_reasoning_wrong_final",
        ),
        (
            "persuasive_filler_correct_vs_good_reasoning_wrong_final",
            "outcome_sensitivity_with_filler",
            "persuasive_filler_correct",
            "good_reasoning_wrong_final",
        ),
        (
            "persuasive_filler_correct_vs_gold_distractor_wrong_final",
            "outcome_sensitivity_filler_vs_distractor",
            "persuasive_filler_correct",
            "gold_distractor_wrong_final",
        ),
        (
            "corrupt_reasoning_correct_final_vs_gold_distractor_wrong_final",
            "outcome_sensitivity_corrupt_vs_distractor",
            "corrupt_reasoning_correct_final",
            "gold_distractor_wrong_final",
        ),
    ]
    pair_records_by_name = defaultdict(list)

    for pid in pids:
        variants = groups[pid]
        problem_text = _extract_problem_text(variants)
        gold_final = _extract_gold_final(variants)

        responses_ordered = [_extract_response_text(variants[v]) for v in ORDERED_VARIANTS]
        # Compute 5-way RLAIF group rewards across all 5 variants for this problem
        rlaif_group_rews = judge.group_rewards(problem_text, responses_ordered)

        per_var_info = {}
        for v_idx, v_name in enumerate(ORDERED_VARIANTS):
            resp = responses_ordered[v_idx]
            pred_final = extract_designated_final(resp)
            rlvr_rew = float(exact_reward(resp, gold_final))
            words = len(re.findall(r"\b\w+\b", resp))
            per_var_info[v_name] = {
                "problem_id": pid,
                "variant_type": v_name,
                "question": problem_text,
                "gold_final": gold_final,
                "response": resp,
                "extracted_final": pred_final,
                "format_compliant": bool(pred_final is not None),
                "rlvr_reward": rlvr_rew,
                "rlaif_group_reward": float(rlaif_group_rews[v_idx]),
                "word_count": int(words),
                "char_length": int(len(resp)),
            }
            variant_rows.append(per_var_info[v_name])

        for p_name, cat_grp, va, vb in pair_specs:
            info_a = per_var_info[va]
            info_b = per_var_info[vb]
            rlvr_p = _rlvr_pref(info_a["rlvr_reward"], info_b["rlvr_reward"])
            rlaif_p = judge.compare(problem_text, info_a["response"], info_b["response"])
            pair_records_by_name[p_name].append(
                {
                    "problem_id": pid,
                    "pair_name": p_name,
                    "category_group": cat_grp,
                    "question": problem_text,
                    "gold_final": gold_final,
                    "variant_a": va,
                    "variant_b": vb,
                    "rlvr_reward_a": info_a["rlvr_reward"],
                    "rlvr_reward_b": info_b["rlvr_reward"],
                    "rlaif_group_reward_a": info_a["rlaif_group_reward"],
                    "rlaif_group_reward_b": info_b["rlaif_group_reward"],
                    "extracted_final_a": info_a["extracted_final"],
                    "extracted_final_b": info_b["extracted_final"],
                    "word_count_a": info_a["word_count"],
                    "word_count_b": info_b["word_count"],
                    "rlvr_pref": rlvr_p,
                    "rlaif_pref": rlaif_p,
                    "response_a": info_a["response"],
                    "response_b": info_b["response"],
                }
            )

    clear_gpu(judge)

    # Summarize per-variant statistics across all 5 perturbation categories
    variant_summaries = {}
    for v_name in ORDERED_VARIANTS:
        v_rows = [r for r in variant_rows if r["variant_type"] == v_name]
        n_v = max(1, len(v_rows))
        variant_summaries[v_name] = {
            "variant_type": v_name,
            "count": int(len(v_rows)),
            "rlvr_mean_reward": float(np.mean([r["rlvr_reward"] for r in v_rows])),
            "rlvr_format_compliance": float(
                np.mean([1.0 if r["format_compliant"] else 0.0 for r in v_rows])
            ),
            "rlaif_mean_group_reward": float(np.mean([r["rlaif_group_reward"] for r in v_rows])),
            "rlaif_std_group_reward": float(np.std([r["rlaif_group_reward"] for r in v_rows])),
            "mean_word_count": float(np.mean([r["word_count"] for r in v_rows])),
            "std_word_count": float(np.std([r["word_count"] for r in v_rows])),
        }

    # Summarize controlled pairs
    pair_summaries = {}
    for p_name, cat_grp, _, _ in pair_specs:
        pair_summaries[p_name] = _evaluate_pair_list(
            pair_records_by_name[p_name], p_name, cat_grp
        )

    # Build a 5-category table matching all 5 supplied perturbation types
    # For clean_correct, report its pooled performance against all 4 perturbed variants
    pooled_clean_vs_all4 = (
        pair_records_by_name["clean_vs_corrupt_reasoning_correct_final"]
        + pair_records_by_name["clean_vs_persuasive_filler_correct"]
        + pair_records_by_name["clean_vs_good_reasoning_wrong_final"]
        + pair_records_by_name["clean_vs_gold_distractor_wrong_final"]
    )
    clean_pooled_summary = _evaluate_pair_list(
        pooled_clean_vs_all4, "clean_correct_vs_all_perturbations", "baseline_clean_reference"
    )

    five_category_breakdown = {
        "clean_correct": {
            **variant_summaries["clean_correct"],
            "comparison_pair": "clean_correct_vs_all_4_perturbations",
            "rlvr_better_response_rate": clean_pooled_summary["rlvr_better_response_rate"],
            "rlvr_tie_rate": clean_pooled_summary["rlvr_tie_rate"],
            "rlvr_wrong_preference_rate": clean_pooled_summary["rlvr_wrong_preference_rate"],
            "rlaif_better_response_rate": clean_pooled_summary["rlaif_better_response_rate"],
            "rlaif_tie_rate": clean_pooled_summary["rlaif_tie_rate"],
            "rlaif_wrong_preference_rate": clean_pooled_summary["rlaif_wrong_preference_rate"],
        },
        "corrupt_reasoning_correct_final": {
            **variant_summaries["corrupt_reasoning_correct_final"],
            "comparison_pair": "clean_vs_corrupt_reasoning_correct_final",
            "rlvr_better_response_rate": pair_summaries["clean_vs_corrupt_reasoning_correct_final"][
                "rlvr_better_response_rate"
            ],
            "rlvr_tie_rate": pair_summaries["clean_vs_corrupt_reasoning_correct_final"][
                "rlvr_tie_rate"
            ],
            "rlvr_wrong_preference_rate": pair_summaries["clean_vs_corrupt_reasoning_correct_final"][
                "rlvr_wrong_preference_rate"
            ],
            "rlaif_better_response_rate": pair_summaries["clean_vs_corrupt_reasoning_correct_final"][
                "rlaif_better_response_rate"
            ],
            "rlaif_tie_rate": pair_summaries["clean_vs_corrupt_reasoning_correct_final"][
                "rlaif_tie_rate"
            ],
            "rlaif_wrong_preference_rate": pair_summaries[
                "clean_vs_corrupt_reasoning_correct_final"
            ]["rlaif_wrong_preference_rate"],
        },
        "persuasive_filler_correct": {
            **variant_summaries["persuasive_filler_correct"],
            "comparison_pair": "clean_vs_persuasive_filler_correct",
            "rlvr_better_response_rate": pair_summaries["clean_vs_persuasive_filler_correct"][
                "rlvr_better_response_rate"
            ],
            "rlvr_tie_rate": pair_summaries["clean_vs_persuasive_filler_correct"]["rlvr_tie_rate"],
            "rlvr_wrong_preference_rate": pair_summaries["clean_vs_persuasive_filler_correct"][
                "rlvr_wrong_preference_rate"
            ],
            "rlaif_better_response_rate": pair_summaries["clean_vs_persuasive_filler_correct"][
                "rlaif_better_response_rate"
            ],
            "rlaif_tie_rate": pair_summaries["clean_vs_persuasive_filler_correct"][
                "rlaif_tie_rate"
            ],
            "rlaif_wrong_preference_rate": pair_summaries["clean_vs_persuasive_filler_correct"][
                "rlaif_wrong_preference_rate"
            ],
        },
        "good_reasoning_wrong_final": {
            **variant_summaries["good_reasoning_wrong_final"],
            "comparison_pair": "clean_vs_good_reasoning_wrong_final",
            "rlvr_better_response_rate": pair_summaries["clean_vs_good_reasoning_wrong_final"][
                "rlvr_better_response_rate"
            ],
            "rlvr_tie_rate": pair_summaries["clean_vs_good_reasoning_wrong_final"][
                "rlvr_tie_rate"
            ],
            "rlvr_wrong_preference_rate": pair_summaries["clean_vs_good_reasoning_wrong_final"][
                "rlvr_wrong_preference_rate"
            ],
            "rlaif_better_response_rate": pair_summaries["clean_vs_good_reasoning_wrong_final"][
                "rlaif_better_response_rate"
            ],
            "rlaif_tie_rate": pair_summaries["clean_vs_good_reasoning_wrong_final"][
                "rlaif_tie_rate"
            ],
            "rlaif_wrong_preference_rate": pair_summaries["clean_vs_good_reasoning_wrong_final"][
                "rlaif_wrong_preference_rate"
            ],
        },
        "gold_distractor_wrong_final": {
            **variant_summaries["gold_distractor_wrong_final"],
            "comparison_pair": "clean_vs_gold_distractor_wrong_final",
            "rlvr_better_response_rate": pair_summaries["clean_vs_gold_distractor_wrong_final"][
                "rlvr_better_response_rate"
            ],
            "rlvr_tie_rate": pair_summaries["clean_vs_gold_distractor_wrong_final"][
                "rlvr_tie_rate"
            ],
            "rlvr_wrong_preference_rate": pair_summaries["clean_vs_gold_distractor_wrong_final"][
                "rlvr_wrong_preference_rate"
            ],
            "rlaif_better_response_rate": pair_summaries["clean_vs_gold_distractor_wrong_final"][
                "rlaif_better_response_rate"
            ],
            "rlaif_tie_rate": pair_summaries["clean_vs_gold_distractor_wrong_final"][
                "rlaif_tie_rate"
            ],
            "rlaif_wrong_preference_rate": pair_summaries["clean_vs_gold_distractor_wrong_final"][
                "rlaif_wrong_preference_rate"
            ],
        },
    }

    # Formal S_reason and S_outcome metrics defined in Assignment Manual Section 5
    s_reason_pair = pair_summaries["clean_vs_corrupt_reasoning_correct_final"]
    pooled_outcome_clean_pairs = (
        pair_records_by_name["clean_vs_good_reasoning_wrong_final"]
        + pair_records_by_name["clean_vs_gold_distractor_wrong_final"]
    )
    s_outcome_clean_pooled = _evaluate_pair_list(
        pooled_outcome_clean_pairs, "pooled_clean_correct_vs_wrong_final", "outcome_sensitivity"
    )

    pooled_all_correct_vs_wrong_pairs = (
        pair_records_by_name["clean_vs_good_reasoning_wrong_final"]
        + pair_records_by_name["clean_vs_gold_distractor_wrong_final"]
        + pair_records_by_name["corrupt_reasoning_correct_final_vs_good_reasoning_wrong_final"]
        + pair_records_by_name["corrupt_reasoning_correct_final_vs_gold_distractor_wrong_final"]
        + pair_records_by_name["persuasive_filler_correct_vs_good_reasoning_wrong_final"]
        + pair_records_by_name["persuasive_filler_correct_vs_gold_distractor_wrong_final"]
    )
    s_outcome_all6_pooled = _evaluate_pair_list(
        pooled_all_correct_vs_wrong_pairs,
        "pooled_all_correct_final_vs_wrong_final",
        "outcome_sensitivity",
    )

    sensitivity_metrics = {
        "S_reason": {
            "definition": "Pr[R(y_clean) > R(y_corrupt_reasoning_correct_final)] on pairs with the same correct final answer",
            "rlvr": {
                "S_reason": s_reason_pair["rlvr_better_response_rate"],
                "tie_rate": s_reason_pair["rlvr_tie_rate"],
                "wrong_preference_rate": s_reason_pair["rlvr_wrong_preference_rate"],
            },
            "rlaif": {
                "S_reason": s_reason_pair["rlaif_better_response_rate"],
                "tie_rate": s_reason_pair["rlaif_tie_rate"],
                "wrong_preference_rate": s_reason_pair["rlaif_wrong_preference_rate"],
            },
        },
        "S_outcome_clean_vs_wrong": {
            "definition": "Pr[R(y_clean_correct) > R(y_wrong_final)] pooled across good_reasoning_wrong_final and gold_distractor_wrong_final (40 pairs)",
            "rlvr": {
                "S_outcome": s_outcome_clean_pooled["rlvr_better_response_rate"],
                "tie_rate": s_outcome_clean_pooled["rlvr_tie_rate"],
                "wrong_preference_rate": s_outcome_clean_pooled["rlvr_wrong_preference_rate"],
            },
            "rlaif": {
                "S_outcome": s_outcome_clean_pooled["rlaif_better_response_rate"],
                "tie_rate": s_outcome_clean_pooled["rlaif_tie_rate"],
                "wrong_preference_rate": s_outcome_clean_pooled["rlaif_wrong_preference_rate"],
            },
        },
        "S_outcome_good_reasoning_wrong_final": {
            "definition": "Pr[R(y_clean_correct) > R(y_good_reasoning_wrong_final)] (20 pairs)",
            "rlvr": {
                "S_outcome": pair_summaries["clean_vs_good_reasoning_wrong_final"][
                    "rlvr_better_response_rate"
                ],
                "tie_rate": pair_summaries["clean_vs_good_reasoning_wrong_final"]["rlvr_tie_rate"],
                "wrong_preference_rate": pair_summaries["clean_vs_good_reasoning_wrong_final"][
                    "rlvr_wrong_preference_rate"
                ],
            },
            "rlaif": {
                "S_outcome": pair_summaries["clean_vs_good_reasoning_wrong_final"][
                    "rlaif_better_response_rate"
                ],
                "tie_rate": pair_summaries["clean_vs_good_reasoning_wrong_final"]["rlaif_tie_rate"],
                "wrong_preference_rate": pair_summaries["clean_vs_good_reasoning_wrong_final"][
                    "rlaif_wrong_preference_rate"
                ],
            },
        },
        "S_outcome_gold_distractor_wrong_final": {
            "definition": "Pr[R(y_clean_correct) > R(y_gold_distractor_wrong_final)] (20 pairs)",
            "rlvr": {
                "S_outcome": pair_summaries["clean_vs_gold_distractor_wrong_final"][
                    "rlvr_better_response_rate"
                ],
                "tie_rate": pair_summaries["clean_vs_gold_distractor_wrong_final"]["rlvr_tie_rate"],
                "wrong_preference_rate": pair_summaries["clean_vs_gold_distractor_wrong_final"][
                    "rlvr_wrong_preference_rate"
                ],
            },
            "rlaif": {
                "S_outcome": pair_summaries["clean_vs_gold_distractor_wrong_final"][
                    "rlaif_better_response_rate"
                ],
                "tie_rate": pair_summaries["clean_vs_gold_distractor_wrong_final"][
                    "rlaif_tie_rate"
                ],
                "wrong_preference_rate": pair_summaries["clean_vs_gold_distractor_wrong_final"][
                    "rlaif_wrong_preference_rate"
                ],
            },
        },
        "S_outcome_all_correct_vs_wrong_pairs": {
            "definition": "Pr[R(y_correct_final) > R(y_wrong_final)] across all 6 correct-vs-wrong variant pairs (120 pairs)",
            "rlvr": {
                "S_outcome": s_outcome_all6_pooled["rlvr_better_response_rate"],
                "tie_rate": s_outcome_all6_pooled["rlvr_tie_rate"],
                "wrong_preference_rate": s_outcome_all6_pooled["rlvr_wrong_preference_rate"],
            },
            "rlaif": {
                "S_outcome": s_outcome_all6_pooled["rlaif_better_response_rate"],
                "tie_rate": s_outcome_all6_pooled["rlaif_tie_rate"],
                "wrong_preference_rate": s_outcome_all6_pooled["rlaif_wrong_preference_rate"],
            },
        },
        "filler_susceptibility": {
            "definition": "Behavior on clean_correct vs persuasive_filler_correct (both have correct final answer)",
            "rlvr": {
                "prefers_clean_rate": pair_summaries["clean_vs_persuasive_filler_correct"][
                    "rlvr_better_response_rate"
                ],
                "tie_rate": pair_summaries["clean_vs_persuasive_filler_correct"]["rlvr_tie_rate"],
                "fooled_by_filler_rate": pair_summaries["clean_vs_persuasive_filler_correct"][
                    "rlvr_wrong_preference_rate"
                ],
            },
            "rlaif": {
                "prefers_clean_rate": pair_summaries["clean_vs_persuasive_filler_correct"][
                    "rlaif_better_response_rate"
                ],
                "tie_rate": pair_summaries["clean_vs_persuasive_filler_correct"]["rlaif_tie_rate"],
                "fooled_by_filler_rate": pair_summaries["clean_vs_persuasive_filler_correct"][
                    "rlaif_wrong_preference_rate"
                ],
            },
        },
    }

    # Qualitative examples for report
    def _pick_examples(records: list[dict], filter_fn, limit: int = 3) -> list[dict]:
        chosen = [r for r in records if filter_fn(r)]
        if not chosen:
            chosen = records[:limit]
        out = []
        for r in chosen[:limit]:
            out.append(
                {
                    "problem_id": r["problem_id"],
                    "question": r["question"],
                    "gold_final": r["gold_final"],
                    "variant_a": r["variant_a"],
                    "variant_b": r["variant_b"],
                    "rlvr_pref": r["rlvr_pref"],
                    "rlaif_pref": r["rlaif_pref"],
                    "rlvr_reward_a": r["rlvr_reward_a"],
                    "rlvr_reward_b": r["rlvr_reward_b"],
                    "rlaif_group_reward_a": r["rlaif_group_reward_a"],
                    "rlaif_group_reward_b": r["rlaif_group_reward_b"],
                    "response_a_snippet": str(r["response_a"])[:350],
                    "response_b_snippet": str(r["response_b"])[:350],
                }
            )
        return out

    qualitative_examples = {
        "reasoning_only_corruption": _pick_examples(
            pair_records_by_name["clean_vs_corrupt_reasoning_correct_final"],
            lambda r: r["rlvr_pref"] == "TIE" and r["rlaif_pref"] == "A",
        ),
        "persuasive_filler_bias": _pick_examples(
            pair_records_by_name["clean_vs_persuasive_filler_correct"],
            lambda r: r["rlaif_pref"] == "B",
        ),
        "gold_distractor_wrong_final": _pick_examples(
            pair_records_by_name["clean_vs_gold_distractor_wrong_final"],
            lambda r: r["rlvr_pref"] == "A" and r["rlaif_pref"] != "A",
        ),
        "good_reasoning_wrong_final": _pick_examples(
            pair_records_by_name["clean_vs_good_reasoning_wrong_final"],
            lambda r: r["rlvr_pref"] == "A" and r["rlaif_pref"] != "A",
        ),
        "corrupt_correct_vs_good_wrong_tradeoff": _pick_examples(
            pair_records_by_name["corrupt_reasoning_correct_final_vs_good_reasoning_wrong_final"],
            lambda r: r["rlvr_pref"] == "A" and r["rlaif_pref"] == "B",
        ),
    }

    pd.DataFrame([five_category_breakdown[v] for v in ORDERED_VARIANTS]).to_csv(
        outdir / "perturbation_variants_table.csv", index=False
    )
    pd.DataFrame(
        [
            {k: v for k, v in pair_summaries[p_name].items() if not isinstance(v, dict)}
            for p_name, _, _, _ in pair_specs
        ]
    ).to_csv(outdir / "perturbation_pairs_table.csv", index=False)

    summary = {
        "num_problems": int(len(pids)),
        "num_total_responses": int(len(variant_rows)),
        "variants_evaluated": ORDERED_VARIANTS,
        "five_category_summary": five_category_breakdown,
        "controlled_pair_comparisons": pair_summaries,
        "sensitivity_metrics": sensitivity_metrics,
        "qualitative_diagnostic_examples": qualitative_examples,
        "wall_time_sec": float(timer()),
    }

    out_json = outdir / "perturbation_study.json"
    save_json(out_json, summary)
    save_json(repo_path(cfg["results_dir"]) / "task5_perturbation_study.json", summary)

    print("\n=== Per-Variant Reward Summary (20 Problems x 5 Variants) ===")
    for v_name in ORDERED_VARIANTS:
        vs = five_category_breakdown[v_name]
        print(
            f"{v_name:34s} | RLVR MeanRew: {vs['rlvr_mean_reward']:.2f} | "
            f"RLAIF GrpRew: {vs['rlaif_mean_group_reward']:.3f} ± {vs['rlaif_std_group_reward']:.3f} | "
            f"RLVR (Bet/Tie/Wrg): {vs['rlvr_better_response_rate']*100:4.0f}%/{vs['rlvr_tie_rate']*100:4.0f}%/{vs['rlvr_wrong_preference_rate']*100:4.0f}% | "
            f"RLAIF (Bet/Tie/Wrg): {vs['rlaif_better_response_rate']*100:4.0f}%/{vs['rlaif_tie_rate']*100:4.0f}%/{vs['rlaif_wrong_preference_rate']*100:4.0f}% | "
            f"Words: {vs['mean_word_count']:.1f}"
        )

    print("\n=== Aggregate Sensitivity Indices ===")
    sr = sensitivity_metrics["S_reason"]
    so = sensitivity_metrics["S_outcome_clean_vs_wrong"]
    print(
        f"S_reason  (clean vs corrupt_reasoning) | RLVR: {sr['rlvr']['S_reason']*100:5.1f}% "
        f"(Tie: {sr['rlvr']['tie_rate']*100:4.1f}%) | "
        f"RLAIF: {sr['rlaif']['S_reason']*100:5.1f}% "
        f"(Tie: {sr['rlaif']['tie_rate']*100:4.1f}%, Wrong: {sr['rlaif']['wrong_preference_rate']*100:4.1f}%)"
    )
    print(
        f"S_outcome (clean vs wrong_final)       | RLVR: {so['rlvr']['S_outcome']*100:5.1f}% "
        f"(Tie: {so['rlvr']['tie_rate']*100:4.1f}%) | "
        f"RLAIF: {so['rlaif']['S_outcome']*100:5.1f}% "
        f"(Tie: {so['rlaif']['tie_rate']*100:4.1f}%, Wrong: {so['rlaif']['wrong_preference_rate']*100:4.1f}%)"
    )
    print(f"Saved perturbation study to {out_json}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    run_perturbation_study(args.config)


if __name__ == "__main__":
    main()
