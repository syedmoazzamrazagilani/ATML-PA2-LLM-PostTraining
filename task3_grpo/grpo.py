from __future__ import annotations

import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    """Return one scalar advantage per sampled completion.

    `group_ids[i]` identifies which prompt produced reward `rewards[i]`.
    Matches the assignment manual definition:
        A_k = (r_k - mu_r) / (sigma_r + eps),  mu_r = (1/K) sum_j r_j
    computed separately within each prompt group. Groups with zero standard
    deviation under tolerance `eps` are uninformative and receive zero advantage.
    """
    rewards_f = rewards.float()
    group_ids_t = torch.as_tensor(group_ids, device=rewards.device)
    adv = torch.zeros_like(rewards_f)
    for g_id in torch.unique(group_ids_t, sorted=True):
        mask = (group_ids_t == g_id)
        group_rewards = rewards_f[mask]
        mean = group_rewards.mean()
        std = group_rewards.std(unbiased=False)
        if float(std.item()) <= eps:
            adv[mask] = 0.0
        else:
            adv[mask] = (group_rewards - mean) / (std + eps)
    return adv.to(dtype=rewards.dtype)


def compute_group_diagnostics(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6) -> dict:
    """Compute within-group reward standard deviation and uninformative-group fraction."""
    rewards_f = rewards.detach().float()
    group_ids_t = torch.as_tensor(group_ids, device=rewards.device)
    group_stds = []
    uninformative = 0
    total_groups = 0
    for g_id in torch.unique(group_ids_t, sorted=True):
        mask = (group_ids_t == g_id)
        group_rewards = rewards_f[mask]
        std = float(group_rewards.std(unbiased=False).item())
        group_stds.append(std)
        total_groups += 1
        if std <= eps:
            uninformative += 1
    mean_std = float(sum(group_stds) / max(1, total_groups))
    uninf_frac = float(uninformative / max(1, total_groups))
    return {
        "within_group_reward_std": mean_std,
        "uninformative_group_fraction": uninf_frac,
        "informative_group_fraction": float(1.0 - uninf_frac),
        "group_stds": group_stds,
    }


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    token_mask = token_mask.to(dtype=new_logp.dtype)
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv.to(dtype=new_logp.dtype)[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    signed_kl = sampled_kl(new_logp.detach(), ref_logp.detach(), token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "signed_sampled_kl": signed_kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
