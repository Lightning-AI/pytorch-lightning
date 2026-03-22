# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

class GradientStatsMonitor(Callback):
    """
    A PyTorch Lightning callback that monitors and logs gradient statistics during training.

    This callback collects gradients after each training batch and computes a set of
    useful metrics to help diagnose training behavior, such as gradient flow, vanishing
    or exploding gradients, and sparsity patterns.

    Features:
        - Logs global gradient norm across all parameters
        - Optionally logs per-layer gradient norms
        - Computes statistical properties of gradients:
            * Mean
            * Standard deviation
        - Measures gradient sparsity (fraction of near-zero values)
        - Detects potential exploding gradients via a configurable threshold
        - Optionally logs gradient histograms for visualization (e.g., TensorBoard)

    Logging Behavior:
        - Metrics are logged every `log_every_n_steps` steps
        - Logging is performed only on the global rank (for distributed training safety)
        - Uses Lightning's `log_dict` for compatibility with all supported loggers

    Args:
        log_every_n_steps (int):
            Frequency (in training steps) at which gradient statistics are logged.

        per_layer (bool):
            If True, logs gradient norms for each parameter individually.
            Parameter names are formatted to be compatible with hierarchical loggers.

        track_stats (bool):
            If True, logs mean and standard deviation of all gradients.

        track_sparsity (bool):
            If True, logs the fraction of gradients that are near zero
            (useful for detecting dead neurons or sparse updates).

        explosion_threshold (float):
            Threshold for the global gradient norm above which a warning is raised,
            indicating potential exploding gradients.

        log_histogram (bool):
            If True, logs the full gradient distribution as a histogram using
            logger backends that support it (e.g., TensorBoard).

    Notes:
        - Parameters with `grad=None` are safely ignored.
        - If no gradients are available (e.g., frozen model), the callback exits silently.
        - Designed to be lightweight and not interfere with the training loop.

    """
    def __init__(
            self,
            log_every_n_steps: int = 1,
            per_layer: bool = False,
            track_stats: bool = True,
            track_sparsity: bool = True,
            explosion_threshold: float = 1e3,
            log_histogram: bool = False,
            
        ):
            super().__init__()
            self.log_every_n_steps = log_every_n_steps
            self.per_layer = per_layer
            self.track_stats = track_stats
            self.track_sparsity = track_sparsity
            self.explosion_threshold = explosion_threshold
            self.log_histogram = log_histogram

    # -------------------------
    # State key
    # -------------------------
    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            log_every_n_steps=self.log_every_n_steps,
            per_layer=self.per_layer,
            track_stats=self.track_stats,
            track_sparsity=self.track_sparsity,
            explosion_threshold=self.explosion_threshold,
            log_histogram=self.log_histogram,
        )
    

    # -------------------------
    # Core hook
    # -------------------------
     
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        EPS = 1e-6
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total_norm = 0.0
        all_grads = []

        layer_norms: Dict[str, float] = {}
        metrics: Dict[str, float] = {}

        # -------------------------
        # Collect gradients
        # -------------------------
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.detach()

            # Flatten for global stats
            all_grads.append(grad.view(-1))

            # Norm
            param_norm = grad.norm(2).item()
            total_norm += param_norm ** 2 # for the global norm over the layers 

            if self.per_layer:
                safe_name = name.replace(".", "/")# Replace "." to make names compatible with hierarchical loggers (e.g., TensorBoard)
                layer_norms[f"grad/{safe_name}_norm"] = param_norm

        if len(all_grads) == 0:
            return
        
        # -------------------------
        # Global norm
        # -------------------------
        total_norm = total_norm ** 0.5
        metrics["grad/global_norm"] = total_norm

        # -------------------------
        # Stack gradients
        # -------------------------
        all_grads_tensor = torch.cat(all_grads)

        # -------------------------
        # Mean / Variance
        # -------------------------
        if self.track_stats:
            metrics["grad/mean"] = all_grads_tensor.mean().item()
            metrics["grad/std"] = all_grads_tensor.std(unbiased=False).item()


        # -------------------------
        # Sparsity (fraction near zero)
        # -------------------------
        if self.track_sparsity:
            zero_fraction = (all_grads_tensor.abs() < EPS).float().mean().item()
            metrics["grad/sparsity"] = zero_fraction

        # -------------------------
        # Explosion warning
        # -------------------------
        if total_norm > self.explosion_threshold:
            rank_zero_warn(
                f"Gradient norm is very high ({total_norm:.2f}). Possible exploding gradients."
            )

        # -------------------------
        # Per-layer norms
        # -------------------------
        if self.per_layer:
            metrics.update(layer_norms)

        # -------------------------
        # Logging (scalars)
        # -------------------------
        if trainer.is_global_zero:
            if trainer.logger is not None:
                pl_module.log_dict(metrics, prog_bar=False, logger=True)

            if self.log_histogram and trainer.logger is not None:
                exp = getattr(trainer.logger, "experiment", None)
                if exp is not None and hasattr(exp, "add_histogram"):
                        exp.add_histogram(
                            "grad/all",
                            all_grads_tensor,
                            global_step=trainer.global_step,
                        )