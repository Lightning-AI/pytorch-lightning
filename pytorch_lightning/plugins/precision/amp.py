import torch
from lightning.pytorch.utilities import rank_zero_warn


def optimizer_step(self, optimizer, model, optimizer_idx, closure, **kwargs):
    """Performs the actual optimizer step with proper gradient scaling."""
    scaler = self.scaler

    # Scale loss and compute gradients
    if closure is not None:
        with torch.cuda.amp.autocast():
            loss = closure()
            scaler.scale(loss).backward()

    try:
        # Unscale gradients before optimizer step
        scaler.unscale_(optimizer)

        # Check if gradients are finite
        valid_gradients = True
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    valid_gradients = False
                    break
            if not valid_gradients:
                break

        if valid_gradients:
            # If gradients are valid, step optimizer and update scaler
            optimizer.step()
            scaler.update()
        else:
            # Skip step and adjust scaler
            scaler.update()
            rank_zero_warn(
                "Gradients have become NaN or inf. Skipping optimizer step but updating scaler. "
                "This may affect model convergence.",
                category=RuntimeWarning,
            )
    except RuntimeError as e:
        if "unscale_() has already been called" not in str(e):
            raise
        # Handle case where unscale was already called
        optimizer.step()
        scaler.update()

    optimizer.zero_grad()
