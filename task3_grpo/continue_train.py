from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean, safe_corr
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import (
    compute_group_diagnostics,
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def _scrub_gradients(model) -> None:
    for p in trainable_parameters(model):
        if p.grad is not None:
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_token_categorical_entropy(logits: torch.Tensor, mask: torch.Tensor) -> float:
    with torch.no_grad():
        tok_ent = torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.float32)
        for t_idx in range(logits.shape[1]):
            step_logits = logits[:, t_idx, :].float()
            probs = torch.softmax(step_logits, dim=-1)
            log_probs = torch.log_softmax(step_logits, dim=-1)
            tok_ent[:, t_idx] = -(probs * log_probs).sum(dim=-1)
        tok_ent = torch.nan_to_num(tok_ent, nan=0.0, posinf=0.0, neginf=0.0)
        return float(masked_mean(tok_ent, mask).item())


def _summarize_length_gradients(completion_records: list[dict]) -> dict:
    valid = [r for r in completion_records if not r["truncated"] and r["length"] > 0]
    if not valid:
        valid = [r for r in completion_records if r["length"] > 0]
    if not valid:
        return {
            "num_valid_completions": 0,
            "median_length": 0.0,
            "short_per_token_grad_mean": 0.0,
            "long_per_token_grad_mean": 0.0,
            "short_to_long_per_token_grad_ratio": 0.0,
            "short_seq_grad_mean": 0.0,
            "long_seq_grad_mean": 0.0,
            "short_effective_token_weight_mean": 0.0,
            "long_effective_token_weight_mean": 0.0,
            "corr_length_per_token_grad": 0.0,
            "corr_length_seq_grad": 0.0,
            "corr_length_effective_weight": 0.0,
        }

    lengths = np.array([r["length"] for r in valid], dtype=float)
    tok_grads = np.array([r["per_token_grad_mag"] for r in valid], dtype=float)
    seq_grads = np.array([r["seq_grad_mag"] for r in valid], dtype=float)
    eff_weights = np.array([r["effective_token_weight"] for r in valid], dtype=float)

    median_len = float(np.median(lengths))
    short_mask = lengths <= median_len
    long_mask = lengths > median_len
    if not np.any(long_mask):
        long_mask = short_mask

    short_tok = float(np.mean(tok_grads[short_mask]))
    long_tok = float(np.mean(tok_grads[long_mask]))
    short_seq = float(np.mean(seq_grads[short_mask]))
    long_seq = float(np.mean(seq_grads[long_mask]))
    short_w = float(np.mean(eff_weights[short_mask]))
    long_w = float(np.mean(eff_weights[long_mask]))

    return {
        "num_valid_completions": int(len(valid)),
        "median_length": median_len,
        "short_mean_length": float(np.mean(lengths[short_mask])),
        "long_mean_length": float(np.mean(lengths[long_mask])),
        "short_per_token_grad_mean": short_tok,
        "long_per_token_grad_mean": long_tok,
        "short_to_long_per_token_grad_ratio": float(short_tok / max(long_tok, 1e-12)),
        "short_seq_grad_mean": short_seq,
        "long_seq_grad_mean": long_seq,
        "short_to_long_seq_grad_ratio": float(short_seq / max(long_seq, 1e-12)),
        "short_effective_token_weight_mean": short_w,
        "long_effective_token_weight_mean": long_w,
        "short_to_long_effective_weight_ratio": float(short_w / max(long_w, 1e-12)),
        "corr_length_per_token_grad": safe_corr(lengths, tok_grads),
        "corr_length_seq_grad": safe_corr(lengths, seq_grads),
        "corr_length_effective_weight": safe_corr(lengths, eff_weights),
    }


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
):
    timer = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)

    total_updates = int(cfg["updates"])
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    k_gen = int(cfg.get("num_generations", 4))
    policy_epochs = int(cfg.get("policy_epochs", 1))
    max_prompt_length = int(cfg.get("max_prompt_length", 256))
    max_completion_length = int(cfg.get("max_completion_length", 512))
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    eps = float(cfg.get("clip_epsilon", 0.20))
    beta = float(cfg.get("kl_beta", 0.10))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    gen_cfg = cfg.get("generation", {})
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))
    do_sample = bool(gen_cfg.get("do_sample", True))

    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)

    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{run_name}_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]

    device = next(policy.parameters()).device

    print(
        f"--- Starting GRPO Continuation: {run_name} | loss_type={loss_type} | "
        f"updates={total_updates} | K={k_gen} | eps={eps} | beta={beta} ---"
    )

    update_logs = []
    all_completion_records = []

    for update_idx in tqdm(range(total_updates), desc=f"GRPO ({run_name})"):
        batch_rows = [
            prompt_rows[(update_idx * prompts_per_update + i) % len(prompt_rows)]
            for i in range(prompts_per_update)
        ]
        prompt_ids = [
            row.get("source_index", row.get("id", (update_idx * prompts_per_update + i) % len(prompt_rows)))
            for i, row in enumerate(batch_rows)
        ]

        prompts_k = []
        group_id_list = []
        for g_idx, row in enumerate(batch_rows):
            msg = prompt_messages(row)
            for _ in range(k_gen):
                prompts_k.append(msg)
                group_id_list.append(g_idx)

        group_ids = torch.tensor(group_id_list, dtype=torch.long, device=device)

        policy.config.use_cache = True
        gen = batch_generate(
            policy,
            tokenizer,
            prompts_k,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_completion_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        policy.config.use_cache = False

        sequences = gen["sequences"].clone()
        attention_mask = gen["attention_mask"].clone()
        prompt_width = gen["prompt_width"]
        response_ids = gen["response_ids"].clone()
        raw_response_mask = gen["response_mask"].clone().to(device)
        responses = gen["responses"]
        truncated = gen["truncated"]
        lengths = gen["response_lengths"]

        rewards = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            prompts_k,
            responses,
            max_length=int(cfg.get("reward_max_length", 1024)),
        ).to(device)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        seq_adv = group_relative_advantages(rewards, group_ids)
        group_diag = compute_group_diagnostics(rewards, group_ids)

        if mask_truncated:
            loss_mask = mask_truncated_sequences(raw_response_mask, truncated)
        else:
            loss_mask = raw_response_mask

        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            old_logp = torch.nan_to_num(old_logp, nan=0.0, posinf=0.0, neginf=0.0).detach().clone()

            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy, sequences, attention_mask, prompt_width, response_ids
                )
                ref_logp = torch.nan_to_num(ref_logp, nan=0.0, posinf=0.0, neginf=0.0).detach().clone()

        policy.train()
        total_loss_val = 0.0
        policy_term_val = 0.0
        sampled_kl_val = 0.0
        signed_kl_val = 0.0
        clip_frac_val = 0.0
        ratio_mean_val = 1.0
        sample_ent_val = 0.0
        token_ent_val = 0.0
        grad_norm_val = 0.0
        per_seq_tok_grad = [0.0] * len(responses)
        per_seq_total_grad = [0.0] * len(responses)

        for _ in range(policy_epochs):
            optimizer.zero_grad()
            new_logp, logits = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            token_ent_val = _compute_token_categorical_entropy(logits, raw_response_mask)
            del logits

            new_logp.retain_grad()
            loss, loss_metrics = grpo_policy_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                seq_adv=seq_adv.detach(),
                token_mask=loss_mask,
                ref_logp=ref_logp,
                eps=eps,
                beta=beta,
                loss_type=loss_type,
                max_completion_length=max_completion_length,
            )

            if float(loss_mask.sum().item()) > 0.0 and torch.isfinite(loss):
                loss.backward()
                if new_logp.grad is not None:
                    grad_mat = torch.nan_to_num(new_logp.grad.detach(), nan=0.0).abs() * loss_mask
                    seq_sums = grad_mat.sum(dim=-1)
                    seq_denoms = loss_mask.sum(dim=-1).clamp_min(1.0)
                    per_seq_total_grad = [float(x.item()) for x in seq_sums]
                    per_seq_tok_grad = [float(x.item()) for x in (seq_sums / seq_denoms)]
                _scrub_gradients(policy)
                grad_norm_val = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters(policy), max_grad_norm
                    ).item()
                )
                optimizer.step()
            else:
                grad_norm_val = 0.0

            total_loss_val = float(loss.detach().item())
            policy_term_val = float(loss_metrics["policy_term"].item())
            sampled_kl_val = float(loss_metrics["sampled_kl"].item())
            signed_kl_val = float(loss_metrics["signed_sampled_kl"].item())
            clip_frac_val = float(loss_metrics["clip_fraction"].item())
            ratio_mean_val = float(loss_metrics["ratio_mean"].item())
            sample_ent_val = float(loss_metrics["sample_entropy"].item())

            del new_logp, loss, loss_metrics

        del old_logp, ref_logp, sequences, attention_mask, response_ids
        clear_gpu()

        n_seq = max(1, len(responses))
        eff_weights = []
        for s_idx in range(len(responses)):
            t_k = max(1, int(lengths[s_idx]))
            if bool(truncated[s_idx]) and mask_truncated:
                w_k = 0.0
            elif loss_type == "grpo":
                w_k = 1.0 / float(n_seq * t_k)
            else:
                w_k = 1.0 / float(n_seq * max_completion_length)
            eff_weights.append(w_k)
            all_completion_records.append(
                {
                    "update": int(update_idx),
                    "source_index": prompt_ids[s_idx // k_gen],
                    "generation_index": int(s_idx % k_gen),
                    "length": int(lengths[s_idx]),
                    "truncated": bool(truncated[s_idx]),
                    "reward": float(rewards[s_idx].item()),
                    "advantage": float(seq_adv[s_idx].item()),
                    "effective_token_weight": float(w_k),
                    "per_token_grad_mag": float(per_seq_tok_grad[s_idx]),
                    "seq_grad_mag": float(per_seq_total_grad[s_idx]),
                }
            )

        peak_vram_gb = (
            float(torch.cuda.max_memory_allocated() / (1024**3))
            if torch.cuda.is_available()
            else 0.0
        )
        elapsed_sec = float(timer())
        lengths_arr = np.asarray(lengths, dtype=float)

        record = {
            "update": int(update_idx),
            "run_name": run_name,
            "loss_type": loss_type,
            "prompt_indices": prompt_ids,
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "rewards": [float(r.item()) for r in rewards],
            "advantages": [float(a.item()) for a in seq_adv],
            "within_group_reward_std": float(group_diag["within_group_reward_std"]),
            "uninformative_group_fraction": float(group_diag["uninformative_group_fraction"]),
            "informative_group_fraction": float(group_diag["informative_group_fraction"]),
            "total_loss": total_loss_val,
            "policy_loss": policy_term_val,
            "sampled_kl": sampled_kl_val,
            "signed_sampled_kl": signed_kl_val,
            "clip_fraction": clip_frac_val,
            "ratio_mean": ratio_mean_val,
            "sample_entropy": sample_ent_val,
            "token_entropy": token_ent_val,
            "grad_norm": grad_norm_val,
            "response_length_mean": float(np.mean(lengths_arr)),
            "response_length_std": float(np.std(lengths_arr)),
            "response_lengths": [int(x) for x in lengths],
            "truncated": [bool(x) for x in truncated],
            "truncated_fraction": float(np.mean(np.asarray(truncated, dtype=float))),
            "effective_token_weights": eff_weights,
            "per_token_grad_mag": per_seq_tok_grad,
            "seq_grad_mag": per_seq_total_grad,
            "peak_vram_gb": peak_vram_gb,
            "elapsed_sec": elapsed_sec,
        }
        append_jsonl(log_path, record)
        update_logs.append(record)

        if update_idx % 5 == 0 or update_idx == total_updates - 1:
            print(
                f"Up {update_idx:02d} | Rew: {record['reward_mean']:.3f} | "
                f"GrpStd: {record['within_group_reward_std']:.3f} | "
                f"Uninf: {record['uninformative_group_fraction']:.2f} | "
                f"PolLoss: {record['policy_loss']:.4f} | KL: {record['sampled_kl']:.4f} | "
                f"Grad: {record['grad_norm']:.3f} | Ent: {record['token_entropy']:.2f} | "
                f"Len: {record['response_length_mean']:.1f} | "
                f"VRAM: {peak_vram_gb:.2f}GB | {elapsed_sec:.1f}s"
            )

    policy.save_pretrained(out)

    all_lengths = [r["length"] for r in all_completion_records]
    len_q75, len_q25 = np.percentile(all_lengths, [75, 25]) if all_lengths else (0.0, 0.0)
    length_grad_summary = _summarize_length_gradients(all_completion_records)

    summary = {
        "run_name": run_name,
        "loss_type": loss_type,
        "updates": total_updates,
        "num_generations_per_prompt": k_gen,
        "prompts_per_update": prompts_per_update,
        "clip_epsilon": eps,
        "kl_beta": beta,
        "learning_rate": float(cfg["learning_rate"]),
        "max_completion_length": max_completion_length,
        "mask_truncated_completions": mask_truncated,
        "seed": int(cfg["seed"]),
        "output_dir": str(out),
        "log_file": str(log_path),
        "prompt_indices_used": [u["prompt_indices"] for u in update_logs],
        "reward_mean": float(np.mean([u["reward_mean"] for u in update_logs])),
        "reward_std_across_updates": float(np.std([u["reward_mean"] for u in update_logs])),
        "reward_initial": float(update_logs[0]["reward_mean"]) if update_logs else 0.0,
        "reward_final": float(update_logs[-1]["reward_mean"]) if update_logs else 0.0,
        "within_group_reward_std_mean": float(np.mean([u["within_group_reward_std"] for u in update_logs])),
        "uninformative_group_fraction_mean": float(np.mean([u["uninformative_group_fraction"] for u in update_logs])),
        "informative_group_fraction_mean": float(np.mean([u["informative_group_fraction"] for u in update_logs])),
        "policy_loss_mean": float(np.mean([u["policy_loss"] for u in update_logs])),
        "total_loss_mean": float(np.mean([u["total_loss"] for u in update_logs])),
        "sampled_kl_mean": float(np.mean([u["sampled_kl"] for u in update_logs])),
        "sampled_kl_final": float(update_logs[-1]["sampled_kl"]) if update_logs else 0.0,
        "signed_sampled_kl_mean": float(np.mean([u["signed_sampled_kl"] for u in update_logs])),
        "signed_sampled_kl_final": float(update_logs[-1]["signed_sampled_kl"]) if update_logs else 0.0,
        "clip_fraction_mean": float(np.mean([u["clip_fraction"] for u in update_logs])),
        "grad_norm_mean": float(np.mean([u["grad_norm"] for u in update_logs])),
        "grad_norm_max": float(np.max([u["grad_norm"] for u in update_logs])) if update_logs else 0.0,
        "sample_entropy_mean": float(np.mean([u["sample_entropy"] for u in update_logs])),
        "token_entropy_mean": float(np.mean([u["token_entropy"] for u in update_logs])),
        "response_length_mean": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "response_length_std": float(np.std(all_lengths)) if all_lengths else 0.0,
        "response_length_iqr": float(len_q75 - len_q25),
        "truncated_fraction_mean": float(np.mean([u["truncated_fraction"] for u in update_logs])),
        "peak_vram_gb": float(max(u["peak_vram_gb"] for u in update_logs)) if update_logs else 0.0,
        "wall_time_sec": float(timer()),
        "length_conditioned_gradient_stats": length_grad_summary,
        "trajectories": {
            "update": [u["update"] for u in update_logs],
            "reward_mean": [u["reward_mean"] for u in update_logs],
            "within_group_reward_std": [u["within_group_reward_std"] for u in update_logs],
            "uninformative_group_fraction": [u["uninformative_group_fraction"] for u in update_logs],
            "policy_loss": [u["policy_loss"] for u in update_logs],
            "total_loss": [u["total_loss"] for u in update_logs],
            "sampled_kl": [u["sampled_kl"] for u in update_logs],
            "signed_sampled_kl": [u["signed_sampled_kl"] for u in update_logs],
            "clip_fraction": [u["clip_fraction"] for u in update_logs],
            "grad_norm": [u["grad_norm"] for u in update_logs],
            "sample_entropy": [u["sample_entropy"] for u in update_logs],
            "token_entropy": [u["token_entropy"] for u in update_logs],
            "response_length_mean": [u["response_length_mean"] for u in update_logs],
            "truncated_fraction": [u["truncated_fraction"] for u in update_logs],
        },
        "completion_records": all_completion_records,
    }

    summary_path = results_dir / f"{run_name}_summary.json"
    save_json(summary_path, summary)
    save_json(repo_path("results") / f"grpo_{run_name}_summary.json", summary)
    print(f"Finished GRPO Training ({run_name}). Adapter saved to {out}")
    print(f"Summary saved to {summary_path}")

    clear_gpu(policy, reward_model, optimizer, bundle)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
