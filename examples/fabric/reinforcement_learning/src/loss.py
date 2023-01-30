import torch


def policy_loss(advantages: torch.Tensor, ratio: torch.Tensor, clip_coef: float) -> torch.Tensor:
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    return pg_loss


def value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_coef: float,
    clip_vloss: bool,
    vf_coef: float,
) -> torch.Tensor:
    new_values = new_values.view(-1)
    if clip_vloss:
        v_loss_unclipped = (new_values - returns) ** 2
        v_clipped = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((new_values - returns) ** 2).mean()
    return v_loss * vf_coef


def entropy_loss(entropy: torch.Tensor, ent_coef: float) -> torch.Tensor:
    return -entropy.mean() * ent_coef
