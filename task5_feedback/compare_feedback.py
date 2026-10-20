from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task5_feedback.evaluate_math import run_math_evaluation
from task5_feedback.score_perturbations import run_perturbation_study


def _load_or_run_eval(cfg: dict, config_path: str, dataset: str) -> dict:
    p = repo_path(cfg["results_dir"]) / "task5_feedback" / f"math_eval_{dataset}.json"
    if p.exists():
        return load_json(p)
    return run_math_evaluation(config_path=config_path, dataset=dataset)


def _load_or_run_perturbations(cfg: dict, config_path: str) -> dict:
    p = repo_path(cfg["results_dir"]) / "task5_feedback" / "perturbation_study.json"
    if p.exists():
        return load_json(p)
    return run_perturbation_study(config_path=config_path)


def run_feedback_comparison(config_path: str = "configs/feedback.yaml") -> dict:
    cfg = load_yaml(config_path)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    gsm = _load_or_run_eval(cfg, config_path, "gsm")
    perturb = _load_or_run_perturbations(cfg, config_path)
    transfer = _load_or_run_eval(cfg, config_path, "transfer")

    policies = ["sft", "rlvr", "rlaif"]
    transfer_comparison = {}
    table_rows = []

    for pol in policies:
        g_pol = gsm["policies"][pol]
        t_pol = transfer["policies"][pol]

        acc_drop = float(t_pol["exact_accuracy"] - g_pol["exact_accuracy"])
        fmt_drop = float(t_pol["format_compliance"] - g_pol["format_compliance"])
        win_drop = float(
            t_pol["ai_pairwise_win_rate_vs_sft"] - g_pol["ai_pairwise_win_rate_vs_sft"]
        )
        len_diff = float(t_pol["response_length_mean"] - g_pol["response_length_mean"])

        row_dict = {
            "policy": pol,
            "gsm_exact_accuracy": float(g_pol["exact_accuracy"]),
            "transfer_exact_accuracy": float(t_pol["exact_accuracy"]),
            "transfer_minus_gsm_accuracy": acc_drop,
            "gsm_format_compliance": float(g_pol["format_compliance"]),
            "transfer_format_compliance": float(t_pol["format_compliance"]),
            "transfer_minus_gsm_format": fmt_drop,
            "gsm_ai_win_rate_vs_sft": float(g_pol["ai_pairwise_win_rate_vs_sft"]),
            "transfer_ai_win_rate_vs_sft": float(t_pol["ai_pairwise_win_rate_vs_sft"]),
            "transfer_minus_gsm_ai_win_rate": win_drop,
            "gsm_wins_ties_losses_vs_sft": f"{g_pol['wins_vs_sft']}/{g_pol['ties_vs_sft']}/{g_pol['losses_vs_sft']}",
            "transfer_wins_ties_losses_vs_sft": f"{t_pol['wins_vs_sft']}/{t_pol['ties_vs_sft']}/{t_pol['losses_vs_sft']}",
            "gsm_verifier_judge_agree_3way": float(g_pol["verifier_judge_agreement_vs_sft_3way"]),
            "transfer_verifier_judge_agree_3way": float(
                t_pol["verifier_judge_agreement_vs_sft_3way"]
            ),
            "gsm_verifier_judge_agree_decisive": float(
                g_pol["verifier_judge_agreement_vs_sft_decisive"]
            ),
            "transfer_verifier_judge_agree_decisive": float(
                t_pol["verifier_judge_agreement_vs_sft_decisive"]
            ),
            "gsm_mean_length": float(g_pol["response_length_mean"]),
            "gsm_std_length": float(g_pol["response_length_std"]),
            "gsm_iqr_length": float(g_pol["response_length_iqr"]),
            "transfer_mean_length": float(t_pol["response_length_mean"]),
            "transfer_std_length": float(t_pol["response_length_std"]),
            "transfer_iqr_length": float(t_pol["response_length_iqr"]),
            "transfer_minus_gsm_length": len_diff,
            "gsm_parsed_wrong_rate": float(g_pol["parsed_wrong_answer_rate"]),
            "transfer_parsed_wrong_rate": float(t_pol["parsed_wrong_answer_rate"]),
            "gsm_format_failure_rate": float(g_pol["format_failure_rate"]),
            "transfer_format_failure_rate": float(t_pol["format_failure_rate"]),
            "gsm_gold_in_text_but_format_failed_rate": float(
                g_pol["gold_in_text_but_format_failed_rate"]
            ),
            "transfer_gold_in_text_but_format_failed_rate": float(
                t_pol["gold_in_text_but_format_failed_rate"]
            ),
        }
        transfer_comparison[pol] = {
            **row_dict,
            "gsm_failure_counts": g_pol.get("failure_type_counts", {}),
            "transfer_failure_counts": t_pol.get("failure_type_counts", {}),
        }
        table_rows.append(row_dict)

    pd.DataFrame(table_rows).to_csv(outdir / "in_vs_out_domain_table.csv", index=False)

    rlaif_vs_rlvr_gsm = gsm["pairwise_comparisons"]["rlaif_vs_rlvr"]
    rlaif_vs_rlvr_transfer = transfer["pairwise_comparisons"]["rlaif_vs_rlvr"]

    sens = perturb["sensitivity_metrics"]
    feedback_tradeoffs = {
        "coverage_and_resolution": {
            "rlvr_s_reason": sens["S_reason"]["rlvr"]["S_reason"],
            "rlaif_s_reason": sens["S_reason"]["rlaif"]["S_reason"],
            "rlvr_s_outcome": sens["S_outcome_clean_vs_wrong"]["rlvr"]["S_outcome"],
            "rlaif_s_outcome": sens["S_outcome_clean_vs_wrong"]["rlaif"]["S_outcome"],
        },
        "noise_and_exploitability": {
            "rlvr_fooled_by_filler_rate": sens["filler_susceptibility"]["rlvr"][
                "fooled_by_filler_rate"
            ],
            "rlaif_fooled_by_filler_rate": sens["filler_susceptibility"]["rlaif"][
                "fooled_by_filler_rate"
            ],
            "rlvr_wrong_on_gold_distractor_rate": sens[
                "S_outcome_gold_distractor_wrong_final"
            ]["rlvr"]["wrong_preference_rate"],
            "rlaif_wrong_on_gold_distractor_rate": sens[
                "S_outcome_gold_distractor_wrong_final"
            ]["rlaif"]["wrong_preference_rate"],
            "rlvr_wrong_on_good_reasoning_wrong_final_rate": sens[
                "S_outcome_good_reasoning_wrong_final"
            ]["rlvr"]["wrong_preference_rate"],
            "rlaif_wrong_on_good_reasoning_wrong_final_rate": sens[
                "S_outcome_good_reasoning_wrong_final"
            ]["rlaif"]["wrong_preference_rate"],
        },
        "compute_and_wall_time": {
            "gsm_eval_wall_time_sec": gsm.get("wall_time_sec", 0.0),
            "transfer_eval_wall_time_sec": transfer.get("wall_time_sec", 0.0),
            "perturbation_study_wall_time_sec": perturb.get("wall_time_sec", 0.0),
        },
    }

    combined = {
        "in_domain_gsm": {
            "num_examples": gsm["num_examples"],
            "policies": gsm["policies"],
            "pairwise_comparisons": gsm["pairwise_comparisons"],
        },
        "out_of_domain_transfer": {
            "num_examples": transfer["num_examples"],
            "policies": transfer["policies"],
            "pairwise_comparisons": transfer["pairwise_comparisons"],
        },
        "in_vs_out_of_domain_comparison": transfer_comparison,
        "direct_rlaif_vs_rlvr": {
            "gsm": rlaif_vs_rlvr_gsm,
            "transfer": rlaif_vs_rlvr_transfer,
        },
        "controlled_perturbation_study": {
            "five_category_summary": perturb["five_category_summary"],
            "controlled_pair_comparisons": perturb["controlled_pair_comparisons"],
            "sensitivity_metrics": perturb["sensitivity_metrics"],
        },
        "feedback_source_tradeoffs": feedback_tradeoffs,
        "qualitative_evidence": {
            "diagnostic_perturbations": perturb["qualitative_diagnostic_examples"],
            "gsm_verifier_judge_disagreements": gsm.get(
                "sample_verifier_judge_disagreements", []
            )[:5],
            "transfer_verifier_judge_disagreements": transfer.get(
                "sample_verifier_judge_disagreements", []
            )[:5],
        },
    }

    out_json = outdir / "feedback_comparison_summary.json"
    save_json(out_json, combined)
    save_json(repo_path(cfg["results_dir"]) / "task5_feedback_summary.json", combined)

    print("\n=====================================================================")
    print("=== Task 5 Final Summary: RLVR vs RLAIF (In-Domain vs Transfer) ===")
    print("=====================================================================")
    for pol in policies:
        tc = transfer_comparison[pol]
        print(
            f"{pol.upper():5s} | GSM Acc: {tc['gsm_exact_accuracy']*100:5.1f}% -> SVAMP Acc: {tc['transfer_exact_accuracy']*100:5.1f}% "
            f"(Delta: {tc['transfer_minus_gsm_accuracy']*100:+5.1f}%) | "
            f"GSM Fmt: {tc['gsm_format_compliance']*100:5.1f}% -> SVAMP Fmt: {tc['transfer_format_compliance']*100:5.1f}% | "
            f"WinRate vs SFT (GSM/SVAMP): {tc['gsm_ai_win_rate_vs_sft']*100:5.1f}%/{tc['transfer_ai_win_rate_vs_sft']*100:5.1f}% | "
            f"Len (GSM/SVAMP): {tc['gsm_mean_length']:5.1f}/{tc['transfer_mean_length']:5.1f}"
        )
    print("\n--- Direct RLAIF vs RLVR Pairwise Comparison ---")
    print(
        f"GSM8K    | RLAIF WinRate: {rlaif_vs_rlvr_gsm['ai_pairwise_win_rate']*100:5.1f}% "
        f"(W:{rlaif_vs_rlvr_gsm['wins']}/T:{rlaif_vs_rlvr_gsm['ties']}/L:{rlaif_vs_rlvr_gsm['losses']}) | "
        f"Ver-Judge Agree (3way/decisive): {rlaif_vs_rlvr_gsm['verifier_judge_agreement_3way']*100:4.1f}%/"
        f"{rlaif_vs_rlvr_gsm['verifier_judge_agreement_on_decisive']*100:4.1f}%"
    )
    print(
        f"SVAMP    | RLAIF WinRate: {rlaif_vs_rlvr_transfer['ai_pairwise_win_rate']*100:5.1f}% "
        f"(W:{rlaif_vs_rlvr_transfer['wins']}/T:{rlaif_vs_rlvr_transfer['ties']}/L:{rlaif_vs_rlvr_transfer['losses']}) | "
        f"Ver-Judge Agree (3way/decisive): {rlaif_vs_rlvr_transfer['verifier_judge_agreement_3way']*100:4.1f}%/"
        f"{rlaif_vs_rlvr_transfer['verifier_judge_agreement_on_decisive']*100:4.1f}%"
    )
    print(f"\nSaved complete Task 5 comparison to {out_json}")
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    run_feedback_comparison(args.config)


if __name__ == "__main__":
    main()
