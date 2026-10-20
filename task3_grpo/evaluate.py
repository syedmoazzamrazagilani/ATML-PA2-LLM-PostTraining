from __future__ import annotations

import argparse
import numpy as np
import torch
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def _token_categorical_entropy_per_row(logits: torch.Tensor, mask: torch.Tensor) -> list[float]:
    with torch.no_grad():
        tok_ent = torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.float32)
        for t_idx in range(logits.shape[1]):
            step_logits = logits[:, t_idx, :].float()
            probs = torch.softmax(step_logits, dim=-1)
            log_probs = torch.log_softmax(step_logits, dim=-1)
            tok_ent[:, t_idx] = -(probs * log_probs).sum(dim=-1)
        tok_ent = torch.nan_to_num(tok_ent, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = (tok_ent * mask).sum(dim=-1)
        row_denoms = mask.sum(dim=-1).clamp_min(1.0)
        return [float(x.item()) for x in (row_sums / row_denoms)]


def evaluate_grpo_policy(
    config_path: str,
    adapter: str,
    name: str = "standard",
    batch_size: int = 4,
) -> dict:
    timer = wall_timer()
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model, reward_tokenizer = bundle["reward"]

    policy.eval()
    policy.config.use_cache = True
    reward_model.eval()

    max_prompt_length = int(cfg.get("max_prompt_length", 256))
    max_completion_length = int(cfg.get("eval_max_completion_length", cfg.get("max_completion_length", 512)))
    gen_cfg = cfg.get("generation", {})
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))
    do_sample = bool(gen_cfg.get("do_sample", True))

    print(
        f"Evaluating GRPO adapter={adapter} ({name}) on {len(rows)} held-out prompts "
        f"(max_new_tokens={max_completion_length})..."
    )

    records = []
    total_valid_tokens = 0.0
    total_k3_kl_tokens = 0.0
    total_signed_kl_tokens = 0.0
    total_sample_ent_tokens = 0.0
    total_cat_ent_tokens = 0.0

    for start in tqdm(range(0, len(rows), batch_size), desc=f"Eval ({name})"):
        chunk = rows[start : start + batch_size]
        prompts = [prompt_messages(r) for r in chunk]

        gen = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_completion_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        sequences = gen["sequences"].clone()
        attention_mask = gen["attention_mask"].clone()
        prompt_width = gen["prompt_width"]
        response_ids = gen["response_ids"].clone()
        response_mask = gen["response_mask"].clone()
        responses = gen["responses"]
        lengths = gen["response_lengths"]
        truncated = gen["truncated"]

        rewards = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            prompts,
            responses,
            max_length=int(cfg.get("reward_max_length", 1024)),
        )
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0).cpu()

        with torch.no_grad():
            pol_logp, logits = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            pol_logp = torch.nan_to_num(pol_logp, nan=0.0, posinf=0.0, neginf=0.0)
            row_cat_ents = _token_categorical_entropy_per_row(logits, response_mask)
            del logits

            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy, sequences, attention_mask, prompt_width, response_ids
                )
                ref_logp = torch.nan_to_num(ref_logp, nan=0.0, posinf=0.0, neginf=0.0)

            log_ratio_ref_over_pol = ref_logp - pol_logp
            per_tok_k3_kl = torch.exp(log_ratio_ref_over_pol) - log_ratio_ref_over_pol - 1.0
            per_tok_signed_kl = pol_logp - ref_logp
            per_tok_sample_ent = -pol_logp

            for idx, row in enumerate(chunk):
                mask_i = response_mask[idx]
                n_tok = float(mask_i.sum().item())
                denom_i = max(1.0, n_tok)

                k3_kl_i = float(((per_tok_k3_kl[idx] * mask_i).sum() / denom_i).item())
                signed_kl_i = float(((per_tok_signed_kl[idx] * mask_i).sum() / denom_i).item())
                seq_kl_i = float((per_tok_signed_kl[idx] * mask_i).sum().item())
                samp_ent_i = float(((per_tok_sample_ent[idx] * mask_i).sum() / denom_i).item())
                cat_ent_i = float(row_cat_ents[idx])

                total_valid_tokens += n_tok
                total_k3_kl_tokens += float((per_tok_k3_kl[idx] * mask_i).sum().item())
                total_signed_kl_tokens += float((per_tok_signed_kl[idx] * mask_i).sum().item())
                total_sample_ent_tokens += float((per_tok_sample_ent[idx] * mask_i).sum().item())
                total_cat_ent_tokens += cat_ent_i * n_tok

                prompt_text = row.get("prompt")
                if not isinstance(prompt_text, str):
                    prompt_text = str(prompts[idx][-1].get("content", ""))

                records.append(
                    {
                        "source_index": row.get("source_index", start + idx),
                        "prompt": prompt_text,
                        "response": responses[idx],
                        "reward": float(rewards[idx].item()),
                        "sampled_kl": k3_kl_i,
                        "signed_sampled_kl": signed_kl_i,
                        "sequence_kl": seq_kl_i,
                        "sample_entropy": samp_ent_i,
                        "token_entropy": cat_ent_i,
                        "length": int(lengths[idx]),
                        "truncated": bool(truncated[idx]),
                    }
                )

        del pol_logp, ref_logp, sequences, attention_mask, response_ids, response_mask
        clear_gpu()

    rewards_arr = np.asarray([r["reward"] for r in records], dtype=float)
    lengths_arr = np.asarray([r["length"] for r in records], dtype=float)
    k3_kls_arr = np.asarray([r["sampled_kl"] for r in records], dtype=float)
    signed_kls_arr = np.asarray([r["signed_sampled_kl"] for r in records], dtype=float)
    seq_kls_arr = np.asarray([r["sequence_kl"] for r in records], dtype=float)
    samp_ents_arr = np.asarray([r["sample_entropy"] for r in records], dtype=float)
    cat_ents_arr = np.asarray([r["token_entropy"] for r in records], dtype=float)
    trunc_arr = np.asarray([r["truncated"] for r in records], dtype=float)

    len_q75, len_q25 = np.percentile(lengths_arr, [75, 25]) if len(lengths_arr) else (0.0, 0.0)
    rew_q75, rew_q25 = np.percentile(rewards_arr, [75, 25]) if len(rewards_arr) else (0.0, 0.0)

    tok_denom = max(1.0, total_valid_tokens)
    summary = {
        "name": name,
        "adapter": str(adapter),
        "num_examples": int(len(records)),
        "reward_mean": float(np.mean(rewards_arr)),
        "reward_std": float(np.std(rewards_arr)),
        "reward_median": float(np.median(rewards_arr)),
        "reward_iqr": float(rew_q75 - rew_q25),
        "sampled_kl_token_mean": float(total_k3_kl_tokens / tok_denom),
        "sampled_kl_seq_avg": float(np.mean(k3_kls_arr)),
        "signed_sampled_kl_token_mean": float(total_signed_kl_tokens / tok_denom),
        "signed_sampled_kl_seq_avg": float(np.mean(signed_kls_arr)),
        "sequence_kl_mean": float(np.mean(seq_kls_arr)),
        "sample_entropy_token_mean": float(total_sample_ent_tokens / tok_denom),
        "sample_entropy_seq_avg": float(np.mean(samp_ents_arr)),
        "token_entropy_token_mean": float(total_cat_ent_tokens / tok_denom),
        "token_entropy_seq_avg": float(np.mean(cat_ents_arr)),
        "response_length_mean": float(np.mean(lengths_arr)),
        "response_length_std": float(np.std(lengths_arr)),
        "response_length_median": float(np.median(lengths_arr)),
        "response_length_iqr": float(len_q75 - len_q25),
        "truncated_fraction": float(np.mean(trunc_arr)),
        "wall_time_sec": float(timer()),
        "examples": records,
    }

    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    out_primary = results_dir / f"{name}_eval.json"
    out_compat = repo_path("results") / f"{name}_eval.json"
    save_json(out_primary, summary)
    save_json(out_compat, summary)

    print(
        f"Eval ({name}) complete | Reward: {summary['reward_mean']:.4f} ± {summary['reward_std']:.4f} | "
        f"KL (k3): {summary['sampled_kl_token_mean']:.4f} | "
        f"KL (signed): {summary['signed_sampled_kl_token_mean']:.4f} | "
        f"Entropy: {summary['token_entropy_token_mean']:.3f} | "
        f"Len: {summary['response_length_mean']:.1f} ± {summary['response_length_std']:.1f} (IQR={summary['response_length_iqr']:.1f})"
    )

    clear_gpu(policy, reward_model, bundle)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", default="outputs/task3_grpo/standard")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    evaluate_grpo_policy(args.config, args.adapter, args.name, args.batch_size)


if __name__ == "__main__":
    main()
