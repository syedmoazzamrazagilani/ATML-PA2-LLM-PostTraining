from __future__ import annotations

import argparse
from collections import Counter
import re
import numpy as np
import pandas as pd

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import clear_gpu, load_policy, load_tokenizer
from scripts.prepare_transfer_eval import materialize_transfer_eval
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final, numerically_equal


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rel_path = dataset_path(cfg, dataset)
    full_path = repo_path(rel_path)
    if dataset == "transfer" and not full_path.exists():
        materialize_transfer_eval(full_path)
    rows = read_jsonl(rel_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def _row_id(row: dict, fallback_idx: int) -> str:
    for k in ("source_index", "prompt_id", "problem_id", "id", "svamp_id"):
        if row.get(k) is not None:
            return str(row[k])
    return str(fallback_idx)


def _row_question(row: dict) -> str:
    for k in ("question", "problem", "prompt"):
        val = row.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", "")).strip()
    return str(row)


def _row_gold_final(row: dict) -> str:
    for k in ("gold_final", "gold", "final_answer", "target"):
        val = row.get(k)
        if val is not None and str(val).strip() != "":
            extracted = extract_designated_final(str(val))
            return extracted if extracted is not None else str(val).strip().replace(",", "")
    ans = row.get("answer")
    if ans is not None:
        extracted = extract_designated_final(str(ans))
        if extracted is not None:
            return extracted
        return str(ans).strip().replace(",", "")
    raise KeyError(f"Could not locate gold_final in row keys={list(row.keys())}")


def _row_prompt_messages(row: dict) -> list[dict]:
    msgs = prompt_messages(row)
    if msgs and isinstance(msgs[-1], dict):
        content = str(msgs[-1].get("content", ""))
        if "####" not in content:
            msgs = [dict(m) for m in msgs]
            msgs[-1]["content"] = (
                content.rstrip()
                + "\n\nShow your reasoning and end your response with exactly `#### <number>`."
            )
    return msgs


def _contains_gold_number_in_text(response: str, gold_final: str) -> bool:
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(response))
    for n_str in nums:
        cleaned = n_str.replace(",", "")
        if numerically_equal(cleaned, gold_final):
            return True
    return False


def generate_policy_responses(
    cfg: dict,
    rows: list[dict],
    tokenizer,
    policy_name: str,
    batch_size: int = 8,
) -> list[dict]:
    set_seed(int(cfg["seed"]))
    model = load_frozen_policy(cfg, policy_name)
    model.eval()
    model.config.use_cache = True

    max_prompt_length = int(cfg.get("max_prompt_length", 256))
    max_new_tokens = int(cfg.get("math_max_new_tokens", 512))

    records = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [_row_prompt_messages(r) for r in chunk]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for offset, (row, resp, n_tok, trunc) in enumerate(
            zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"])
        ):
            idx = start + offset
            sid = _row_id(row, idx)
            question = _row_question(row)
            gold = _row_gold_final(row)
            pred_final = extract_designated_final(resp)
            format_ok = bool(pred_final is not None)
            exact_ok = float(exact_reward(resp, gold))
            gold_in_text = _contains_gold_number_in_text(resp, gold)
            if exact_ok == 1.0:
                failure_type = "correct"
            elif not format_ok:
                failure_type = (
                    "format_failure_gold_in_text" if gold_in_text else "format_failure_no_gold"
                )
            else:
                failure_type = "wrong_final_answer"

            records.append(
                {
                    "source_index": sid,
                    "dataset_index": idx,
                    "policy": policy_name,
                    "question": question,
                    "gold_final": gold,
                    "response": resp,
                    "extracted_final": pred_final,
                    "format_compliant": format_ok,
                    "exact_reward": exact_ok,
                    "gold_in_text": bool(gold_in_text),
                    "failure_type": failure_type,
                    "response_tokens": int(n_tok),
                    "truncated": bool(trunc),
                }
            )

    clear_gpu(model)
    return records


def summarize_policy_records(records: list[dict], policy_name: str) -> dict:
    n = max(1, len(records))
    acc = sum(float(r["exact_reward"]) for r in records) / n
    fmt = sum(1.0 if r["format_compliant"] else 0.0 for r in records) / n
    parsed_wrong = sum(
        1.0 if (r["format_compliant"] and float(r["exact_reward"]) == 0.0) else 0.0
        for r in records
    ) / n
    fmt_fail = 1.0 - fmt
    gold_in_text_rate = sum(1.0 if r["gold_in_text"] else 0.0 for r in records) / n
    gold_in_text_but_fmt_fail = sum(
        1.0 if (not r["format_compliant"] and r["gold_in_text"]) else 0.0 for r in records
    ) / n

    lens = np.asarray([int(r["response_tokens"]) for r in records], dtype=float)
    truncs = np.asarray([1.0 if r["truncated"] else 0.0 for r in records], dtype=float)
    q75, q25 = np.percentile(lens, [75, 25]) if len(lens) else (0.0, 0.0)

    failure_counts = Counter(r["failure_type"] for r in records)

    return {
        "policy": policy_name,
        "num_examples": int(len(records)),
        "exact_accuracy": float(acc),
        "format_compliance": float(fmt),
        "parsed_wrong_answer_rate": float(parsed_wrong),
        "format_failure_rate": float(fmt_fail),
        "gold_in_text_rate": float(gold_in_text_rate),
        "gold_in_text_but_format_failed_rate": float(gold_in_text_but_fmt_fail),
        "response_length_mean": float(np.mean(lens)) if len(lens) else 0.0,
        "response_length_std": float(np.std(lens)) if len(lens) else 0.0,
        "response_length_median": float(np.median(lens)) if len(lens) else 0.0,
        "response_length_iqr": float(q75 - q25),
        "truncated_fraction": float(np.mean(truncs)) if len(truncs) else 0.0,
        "failure_type_counts": dict(failure_counts),
    }


def _verifier_preference(rew_a: float, rew_b: float) -> str:
    if rew_a > rew_b:
        return "A"
    if rew_b > rew_a:
        return "B"
    return "TIE"


def evaluate_pairwise_comparison(
    judge: PairwiseAIJudge,
    records_a: list[dict],
    records_b: list[dict],
    name_a: str,
    name_b: str,
) -> tuple[dict, list[dict]]:
    assert len(records_a) == len(records_b)
    n = max(1, len(records_a))

    wins = 0
    ties = 0
    losses = 0
    exact_3way_agree = 0
    decisive_total = 0
    decisive_strict_agree = 0
    decisive_non_contra = 0
    judge_prefers_wrong = 0

    confusion = {
        v_lbl: {"A": 0, "B": 0, "TIE": 0} for v_lbl in ("A", "B", "TIE")
    }
    pair_rows = []

    for ra, rb in zip(records_a, records_b):
        problem = ra["question"]
        if name_a == name_b and ra["response"] == rb["response"]:
            j_pref = "TIE"
        else:
            j_pref = judge.compare(problem, ra["response"], rb["response"])

        v_pref = _verifier_preference(float(ra["exact_reward"]), float(rb["exact_reward"]))
        confusion[v_pref][j_pref] += 1

        if j_pref == "A":
            wins += 1
            score_a = 1.0
        elif j_pref == "B":
            losses += 1
            score_a = 0.0
        else:
            ties += 1
            score_a = 0.5

        if j_pref == v_pref:
            exact_3way_agree += 1

        if v_pref in {"A", "B"}:
            decisive_total += 1
            if j_pref == v_pref:
                decisive_strict_agree += 1
                decisive_non_contra += 1
            elif j_pref == "TIE":
                decisive_non_contra += 1
            else:
                judge_prefers_wrong += 1

        pair_rows.append(
            {
                "source_index": ra["source_index"],
                "question": problem,
                "gold_final": ra["gold_final"],
                "policy_a": name_a,
                "policy_b": name_b,
                "exact_reward_a": float(ra["exact_reward"]),
                "exact_reward_b": float(rb["exact_reward"]),
                "extracted_final_a": ra["extracted_final"],
                "extracted_final_b": rb["extracted_final"],
                "length_a": int(ra["response_tokens"]),
                "length_b": int(rb["response_tokens"]),
                "verifier_pref": v_pref,
                "judge_pref": j_pref,
                "pairwise_score_a": score_a,
                "verifier_judge_agree_3way": bool(j_pref == v_pref),
            }
        )

    # Flush judge cache at end of comparison
    judge.cache_path.parent.mkdir(parents=True, exist_ok=True)
    judge.cache_path.write_text(
        __import__("json").dumps(judge.cache, indent=2), encoding="utf-8"
    )

    win_rate = float((wins + 0.5 * ties) / n)
    ver_win_rate = float(
        sum(
            1.0
            if r["verifier_pref"] == "A"
            else (0.5 if r["verifier_pref"] == "TIE" else 0.0)
            for r in pair_rows
        )
        / n
    )

    summary = {
        "comparison": f"{name_a}_vs_{name_b}",
        "policy_a": name_a,
        "policy_b": name_b,
        "num_examples": int(len(records_a)),
        "ai_pairwise_win_rate": win_rate,
        "verifier_pairwise_win_rate": ver_win_rate,
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "win_rate_strict": float(wins / n),
        "tie_rate": float(ties / n),
        "loss_rate": float(losses / n),
        "verifier_judge_agreement_3way": float(exact_3way_agree / n),
        "decisive_pairs_count": int(decisive_total),
        "verifier_judge_agreement_on_decisive": float(
            decisive_strict_agree / max(1, decisive_total)
        ),
        "verifier_judge_non_contradiction_on_decisive": float(
            decisive_non_contra / max(1, decisive_total)
        ),
        "judge_prefers_wrong_over_correct_rate_on_decisive": float(
            judge_prefers_wrong / max(1, decisive_total)
        ),
        "confusion_verifier_rows_vs_judge_cols": confusion,
    }
    return summary, pair_rows


def run_math_evaluation(
    config_path: str = "configs/feedback.yaml",
    dataset: str = "gsm",
    batch_size: int = 8,
    reuse_generated: bool = False,
) -> dict:
    timer = wall_timer()
    cfg, rows, tokenizer = load_math_evaluation(config_path, dataset)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    policies = list(policy_specs(cfg).keys())
    print(f"=== Task 5 Math Evaluation ({dataset}) | Rows: {len(rows)} | Policies: {policies} ===")

    records_by_policy = {}
    policy_summaries = {}

    for policy_name in policies:
        gen_path = outdir / f"generated_{dataset}_{policy_name}.jsonl"
        if reuse_generated and gen_path.exists():
            print(f"Reusing cached generations from {gen_path}")
            recs = read_jsonl(gen_path)
        else:
            print(f"\n--- Generating {dataset} responses for policy: {policy_name} ---")
            recs = generate_policy_responses(
                cfg, rows, tokenizer, policy_name, batch_size=batch_size
            )
            write_jsonl(gen_path, recs)
        records_by_policy[policy_name] = recs
        policy_summaries[policy_name] = summarize_policy_records(recs, policy_name)
        ps = policy_summaries[policy_name]
        print(
            f"{policy_name.upper():5s} | ExactAcc: {ps['exact_accuracy']*100:5.1f}% | "
            f"Format: {ps['format_compliance']*100:5.1f}% | "
            f"Len: {ps['response_length_mean']:5.1f} ± {ps['response_length_std']:4.1f}"
        )

    clear_gpu(tokenizer)

    # Load frozen 3B PairwiseAIJudge after all policy models are unloaded from GPU
    cache_path = outdir / f"pairwise_judge_cache_{dataset}.json"
    print(f"\n--- Running PairwiseAIJudge ({cfg['ai_judge_model']}) for dataset={dataset} ---")
    judge = PairwiseAIJudge(cfg, cache_path=cache_path)

    rlvr_vs_sft, pairs_rlvr_sft = evaluate_pairwise_comparison(
        judge, records_by_policy["rlvr"], records_by_policy["sft"], "rlvr", "sft"
    )
    rlaif_vs_sft, pairs_rlaif_sft = evaluate_pairwise_comparison(
        judge, records_by_policy["rlaif"], records_by_policy["sft"], "rlaif", "sft"
    )
    rlaif_vs_rlvr, pairs_rlaif_rlvr = evaluate_pairwise_comparison(
        judge, records_by_policy["rlaif"], records_by_policy["rlvr"], "rlaif", "rlvr"
    )

    # Also compute 3-way RLAIF group rewards across [sft, rlvr, rlaif] using the cached pairwise decisions
    group_rewards_by_policy = {"sft": [], "rlvr": [], "rlaif": []}
    for i in range(len(rows)):
        prob = records_by_policy["sft"][i]["question"]
        resps = [
            records_by_policy["sft"][i]["response"],
            records_by_policy["rlvr"][i]["response"],
            records_by_policy["rlaif"][i]["response"],
        ]
        g_rews = judge.group_rewards(prob, resps)
        group_rewards_by_policy["sft"].append(float(g_rews[0]))
        group_rewards_by_policy["rlvr"].append(float(g_rews[1]))
        group_rewards_by_policy["rlaif"].append(float(g_rews[2]))

    clear_gpu(judge)

    # Attach pairwise metrics against SFT directly to each policy summary
    sft_self_summary = {
        "ai_pairwise_win_rate_vs_sft": 0.5,
        "wins_vs_sft": 0,
        "ties_vs_sft": int(len(rows)),
        "losses_vs_sft": 0,
        "verifier_judge_agreement_vs_sft_3way": 1.0,
        "verifier_judge_agreement_vs_sft_decisive": 1.0,
        "mean_3way_group_reward": float(np.mean(group_rewards_by_policy["sft"])),
    }
    policy_summaries["sft"].update(sft_self_summary)

    for pol_name, comp in [("rlvr", rlvr_vs_sft), ("rlaif", rlaif_vs_sft)]:
        policy_summaries[pol_name].update(
            {
                "ai_pairwise_win_rate_vs_sft": comp["ai_pairwise_win_rate"],
                "wins_vs_sft": comp["wins"],
                "ties_vs_sft": comp["ties"],
                "losses_vs_sft": comp["losses"],
                "verifier_judge_agreement_vs_sft_3way": comp["verifier_judge_agreement_3way"],
                "verifier_judge_agreement_vs_sft_decisive": comp[
                    "verifier_judge_agreement_on_decisive"
                ],
                "verifier_judge_non_contradiction_vs_sft_decisive": comp[
                    "verifier_judge_non_contradiction_on_decisive"
                ],
                "mean_3way_group_reward": float(np.mean(group_rewards_by_policy[pol_name])),
            }
        )

    # Save tabular CSV
    df_table = pd.DataFrame(
        [
            {k: v for k, v in policy_summaries[p].items() if k != "failure_type_counts"}
            for p in policies
        ]
    )
    df_table.to_csv(outdir / f"math_eval_{dataset}_table.csv", index=False)

    # Collect qualitative disagreement examples between verifier and AI judge
    disagreement_examples = []
    for pr, rec_rlvr, rec_rlaif, rec_sft in zip(
        pairs_rlaif_rlvr,
        records_by_policy["rlvr"],
        records_by_policy["rlaif"],
        records_by_policy["sft"],
    ):
        if pr["verifier_pref"] in {"A", "B"} and pr["judge_pref"] != pr["verifier_pref"]:
            disagreement_examples.append(
                {
                    "source_index": pr["source_index"],
                    "question": pr["question"],
                    "gold_final": pr["gold_final"],
                    "verifier_pref_rlaif_vs_rlvr": pr["verifier_pref"],
                    "judge_pref_rlaif_vs_rlvr": pr["judge_pref"],
                    "rlvr_exact": rec_rlvr["exact_reward"],
                    "rlaif_exact": rec_rlaif["exact_reward"],
                    "sft_exact": rec_sft["exact_reward"],
                    "rlvr_extracted": rec_rlvr["extracted_final"],
                    "rlaif_extracted": rec_rlaif["extracted_final"],
                    "rlvr_tokens": rec_rlvr["response_tokens"],
                    "rlaif_tokens": rec_rlaif["response_tokens"],
                    "rlvr_snippet": str(rec_rlvr["response"])[:350],
                    "rlaif_snippet": str(rec_rlaif["response"])[:350],
                }
            )

    result = {
        "dataset": dataset,
        "dataset_path": str(dataset_path(cfg, dataset)),
        "num_examples": int(len(rows)),
        "policies": policy_summaries,
        "pairwise_comparisons": {
            "rlvr_vs_sft": rlvr_vs_sft,
            "rlaif_vs_sft": rlaif_vs_sft,
            "rlaif_vs_rlvr": rlaif_vs_rlvr,
        },
        "sample_verifier_judge_disagreements": disagreement_examples[:10],
        "wall_time_sec": float(timer()),
    }

    out_json = outdir / f"math_eval_{dataset}.json"
    save_json(out_json, result)
    save_json(repo_path(cfg["results_dir"]) / f"task5_math_eval_{dataset}.json", result)
    write_jsonl(outdir / f"pairwise_rlaif_vs_rlvr_{dataset}.jsonl", pairs_rlaif_rlvr)
    write_jsonl(outdir / f"pairwise_rlvr_vs_sft_{dataset}.jsonl", pairs_rlvr_sft)
    write_jsonl(outdir / f"pairwise_rlaif_vs_sft_{dataset}.jsonl", pairs_rlaif_sft)

    print(f"\n=== Summary: Task 5 Math Evaluation ({dataset}) ===")
    for p in policies:
        ps = policy_summaries[p]
        print(
            f"{p.upper():5s} | Acc: {ps['exact_accuracy']*100:5.1f}% | "
            f"Format: {ps['format_compliance']*100:5.1f}% | "
            f"WinRate vs SFT: {ps['ai_pairwise_win_rate_vs_sft']*100:5.1f}% "
            f"(W:{ps['wins_vs_sft']}/T:{ps['ties_vs_sft']}/L:{ps['losses_vs_sft']}) | "
            f"Ver-Judge Agree (3way/decisive): {ps['verifier_judge_agreement_vs_sft_3way']*100:4.1f}%/"
            f"{ps['verifier_judge_agreement_vs_sft_decisive']*100:4.1f}% | "
            f"Len: {ps['response_length_mean']:5.1f}"
        )
    print(
        f"Direct RLAIF vs RLVR | RLAIF WinRate: {rlaif_vs_rlvr['ai_pairwise_win_rate']*100:5.1f}% "
        f"(W:{rlaif_vs_rlvr['wins']}/T:{rlaif_vs_rlvr['ties']}/L:{rlaif_vs_rlvr['losses']}) | "
        f"Ver-Judge Agree (3way/decisive): {rlaif_vs_rlvr['verifier_judge_agreement_3way']*100:4.1f}%/"
        f"{rlaif_vs_rlvr['verifier_judge_agreement_on_decisive']*100:4.1f}%"
    )
    print(f"Saved {out_json}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--reuse-generated", action="store_true")
    args = ap.parse_args()
    run_math_evaluation(
        config_path=args.config,
        dataset=args.dataset,
        batch_size=args.batch_size,
        reuse_generated=args.reuse_generated,
    )


if __name__ == "__main__":
    main()
