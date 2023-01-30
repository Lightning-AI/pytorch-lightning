import torch
import torch.nn.functional as F


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
    if not clip_vloss:
        values_pred = new_values
    else:
        values_pred = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
    return vf_coef * F.mse_loss(values_pred, returns)


def entropy_loss(entropy: torch.Tensor, ent_coef: float) -> torch.Tensor:
    return -entropy.mean() * ent_coef
