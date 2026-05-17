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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from typing_extensions import override

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

if TYPE_CHECKING:
    import lightning.pytorch as pl

_EPS = 1e-6


class GradientStatsMonitor(Callback):
    """A PyTorch Lightning callback that monitors and logs gradient statistics during training.

    Gradients are captured in ``on_before_optimizer_step``, i.e. **before** the Trainer applies
    gradient clipping, so all metrics reflect true unclipped gradients.

    Features:
        - Logs global gradient norm across all parameters
        - Optionally logs per-layer gradient norms
        - Computes mean and standard deviation of gradients
        - Measures gradient sparsity (fraction of near-zero values)
        - Detects potential exploding gradients via a configurable threshold

    Logging Behavior:
        - Per-step metrics are logged under ``train/grad/``
          (e.g. ``train/grad/global_norm``) every ``log_every_n_steps`` global steps.
        - Per-epoch metrics are logged at the end of every epoch under ``train/epoch/grad/``,
          aggregated over all optimizer steps regardless of ``log_every_n_steps``.
        - Logging is performed only on the global rank (for distributed training safety).
        - Uses Lightning's ``log_dict`` for compatibility with all supported loggers.
        - The epoch accumulator and step counter are saved in checkpoints via ``state_dict`` /
          ``load_state_dict``, so epoch aggregates remain correct after a mid-epoch resume.

    Subclassing:
        Override any of the following methods to customise what is computed or logged:

        - ``compute_batch_stats`` — metrics logged after each optimizer step
        - ``init_epoch_stats`` — initial accumulator state at the start of each epoch
        - ``update_epoch_stats`` — how each step updates the accumulator
        - ``compute_epoch_stats`` — metrics logged at the end of each epoch

    Args:
        log_every_n_steps (int):
            Frequency (in global steps) at which per-step gradient statistics are logged,
            i.e. when ``trainer.global_step % log_every_n_steps == 0``.
            Set to ``0`` to disable per-step logging entirely (epoch logging unaffected).

        track_epochs (bool):
            If True, logs gradient statistics aggregated over each full epoch.

        per_layer (bool):
            If True, logs gradient norms for each parameter individually.
            Parameter names are formatted to be compatible with hierarchical loggers.

        track_sparsity (bool):
            If True, logs the fraction of gradients that are near zero
            (useful for detecting dead neurons or sparse updates).

        explosion_threshold (float | None):
            Threshold for the global gradient norm above which a warning is raised.
            Operates on **pre-clip** gradients, so it fires even when ``gradient_clip_val``
            is set. Set to ``None`` to disable.

    Notes:
        - With multiple optimizers, only the first ``on_before_optimizer_step`` call per
          global step is processed; subsequent calls for the same step are skipped.
        - Parameters with ``grad=None`` are safely ignored.
        - If no gradients are available (e.g., frozen model or inside no_grad), the callback
          exits silently.
        - Designed to be lightweight and not interfere with the training loop.

    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        track_epochs: bool = True,
        per_layer: bool = False,
        track_sparsity: bool = True,
        explosion_threshold: float | None = 1e4,
    ):
        super().__init__()
        if not track_epochs and log_every_n_steps <= 0:
            raise ValueError("GradientStatsMonitor logs nothing: set log_every_n_steps > 0 or track_epochs=True.")
        self.log_every_n_steps = log_every_n_steps
        self.track_epochs = track_epochs
        self.per_layer = per_layer
        self.track_sparsity = track_sparsity
        self.explosion_threshold = explosion_threshold
        self._train_stats: dict[str, Any] = self.init_epoch_stats("train")
        self._last_logged_step: int = -1

    # -------------------------
    # Internal helpers
    # -------------------------

    def _on_grad_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        should_log: bool,
    ) -> None:
        layer_grads = self._collect_grads(pl_module)
        if layer_grads is None:
            return

        if self.track_epochs:
            self.update_epoch_stats(self._train_stats, layer_grads)

        threshold = self.explosion_threshold
        if not should_log and threshold is None:
            return

        if should_log:
            # Compute full stats once; reuse the global norm for the explosion check
            # to avoid iterating over all parameters a second time.
            metrics = self.compute_batch_stats("train", layer_grads)
            if threshold is not None:
                norm = metrics.get("train/grad/global_norm")
                if norm is not None and norm > threshold:
                    rank_zero_warn(f"Gradient norm is very high ({norm:.2f}). Possible exploding gradients.")
            self._log_scalars(trainer, pl_module, metrics)
        else:
            # Explosion check only — compute just the norm, not the full stat set.
            norm = sum(g.norm(2).item() ** 2 for g in layer_grads.values()) ** 0.5
            if threshold is not None and norm > threshold:
                rank_zero_warn(f"Gradient norm is very high ({norm:.2f}). Possible exploding gradients.")

    def _collect_grads(self, pl_module: pl.LightningModule) -> dict[str, torch.Tensor] | None:
        """Collect per-layer gradients.

        Returns ``{param_name: flat_grad_tensor}`` for every parameter that has a gradient,
        or ``None`` if no parameter had a gradient this step.

        Gradients are moved to CPU and detached from the graph to avoid memory leaks and
        keep GPU memory free during monitoring.  The returned tensors are flattened for
        easier norm/stat computations.

        """
        layer_grads = {
            name: param.grad.detach().cpu().view(-1)
            for name, param in pl_module.named_parameters()
            if param.grad is not None
        }
        return layer_grads or None

    def _log_scalars(self, trainer: pl.Trainer, pl_module: pl.LightningModule, metrics: dict[str, float]) -> None:
        if trainer.is_global_zero and trainer.logger is not None:
            pl_module.log_dict(metrics, prog_bar=False, logger=True)

    # -------------------------
    # Per-step stats  ->  override to control what is computed from each step's gradients
    # -------------------------

    def compute_batch_stats(self, phase: str, layer_grads: dict[str, torch.Tensor]) -> dict[str, float]:
        """Compute and return the metric dict logged after each optimizer step.

        The returned dict is passed directly to ``pl_module.log_dict``.
        Override to add, remove, or rename metrics.

        """
        p = f"{phase}/"
        total_norm_sq = 0.0
        total_count = 0
        total_sum = 0.0
        total_sq_sum = 0.0
        near_zero_count = 0
        metrics: dict[str, float] = {}
        for name, grad in layer_grads.items():
            norm = grad.norm(2).item()
            total_norm_sq += norm**2
            n = grad.numel()
            total_count += n
            total_sum += grad.sum().item()
            total_sq_sum += grad.pow(2).sum().item()
            if self.track_sparsity:
                near_zero_count += int((grad.abs() < _EPS).sum().item())
            if self.per_layer:
                metrics[f"{p}grad/{name.replace('.', '/')}_norm"] = norm
        metrics[f"{p}grad/global_norm"] = total_norm_sq**0.5
        mean = total_sum / total_count
        variance = total_sq_sum / total_count - mean**2
        metrics[f"{p}grad/mean"] = mean
        metrics[f"{p}grad/std"] = max(variance, 0.0) ** 0.5
        if self.track_sparsity:
            metrics[f"{p}grad/sparsity"] = near_zero_count / total_count
        return metrics

    # -------------------------
    # Per-epoch stats  ->  override to control accumulation and final epoch metrics
    # -------------------------

    def init_epoch_stats(self, phase: str) -> dict[str, Any]:
        """Return a fresh accumulator for the start of an epoch.

        Override to add extra fields that ``update_epoch_stats`` and
        ``compute_epoch_stats`` can then use.

        """
        return {
            "norm_sum": 0.0,
            "grad_sum": 0.0,
            "grad_sq_sum": 0.0,
            "grad_count": 0,
            "near_zero_count": 0,
            "steps": 0,
            "layer_norm_sums": {},
        }

    def update_epoch_stats(self, state: dict[str, Any], layer_grads: dict[str, torch.Tensor]) -> None:
        """Update the epoch accumulator in-place with one step's gradients.

        Override to accumulate additional fields introduced in ``init_epoch_stats``.

        """
        total_norm_sq = 0.0
        for name, grad in layer_grads.items():
            norm = grad.norm(2).item()
            total_norm_sq += norm**2
            state["grad_count"] += grad.numel()
            state["grad_sum"] += grad.sum().item()
            state["grad_sq_sum"] += grad.pow(2).sum().item()
            if self.track_sparsity:
                state["near_zero_count"] += int((grad.abs() < _EPS).sum().item())
            if self.per_layer:
                key = f"grad/{name.replace('.', '/')}_norm"
                state["layer_norm_sums"][key] = state["layer_norm_sums"].get(key, 0.0) + norm
        state["norm_sum"] += total_norm_sq**0.5
        state["steps"] += 1

    def compute_epoch_stats(self, phase: str, state: dict[str, Any]) -> dict[str, float] | None:
        """Compute and return the metric dict logged at the end of each epoch.

        Args:
            phase: the phase prefix used in metric names (currently always ``"train"``).
            state: the accumulator produced by ``init_epoch_stats`` and updated
                by ``update_epoch_stats``.

        Returns ``None`` if no steps were recorded (empty dataloader).
        The returned dict is passed directly to ``pl_module.log_dict``.
        Override to add, remove, or rename metrics, or to derive additional
        values from extra state added in ``init_epoch_stats`` / ``update_epoch_stats``.

        Note:
            ``{phase}/epoch/grad/global_norm`` is the **mean of per-step global norms**
            (i.e. ``mean(‖g_t‖₂)`` over optimizer steps *t*), not the true L2 norm of all
            gradients accumulated over the epoch.  The same applies to per-layer norm averages.

        """
        if state["steps"] == 0:
            return None
        p = f"{phase}/epoch/"
        metrics: dict[str, float] = {f"{p}grad/global_norm": state["norm_sum"] / state["steps"]}
        if state["grad_count"] > 0:
            mean = state["grad_sum"] / state["grad_count"]
            variance = state["grad_sq_sum"] / state["grad_count"] - mean**2
            metrics[f"{p}grad/mean"] = mean
            metrics[f"{p}grad/std"] = max(variance, 0.0) ** 0.5
        if self.track_sparsity and state["grad_count"] > 0:
            metrics[f"{p}grad/sparsity"] = state["near_zero_count"] / state["grad_count"]
        if self.per_layer:
            for key, norm_sum in state["layer_norm_sums"].items():
                metrics[f"{p}{key}"] = norm_sum / state["steps"]
        return metrics

    # -------------------------
    # Hooks
    # -------------------------

    @override
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._train_stats = self.init_epoch_stats("train")

    @override
    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Any) -> None:
        # Guard against multiple calls per step when there are multiple optimizers.
        if trainer.global_step == self._last_logged_step:
            return
        self._last_logged_step = trainer.global_step
        should_log = self.log_every_n_steps > 0 and trainer.global_step % self.log_every_n_steps == 0
        self._on_grad_step(trainer, pl_module, should_log=should_log)

    @override
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.track_epochs:
            return
        metrics = self.compute_epoch_stats("train", self._train_stats)
        if metrics:
            self._log_scalars(trainer, pl_module, metrics)

    # -------------------------
    # Checkpoint save / restore
    # -------------------------

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "train_stats": self._train_stats,
            "last_logged_step": self._last_logged_step,
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._train_stats = state_dict["train_stats"]
        self._last_logged_step = state_dict["last_logged_step"]

    # -------------------------
    # State key
    # -------------------------

    @property
    @override
    def state_key(self) -> str:
        return self._generate_state_key(
            log_every_n_steps=self.log_every_n_steps,
            track_epochs=self.track_epochs,
            per_layer=self.per_layer,
            track_sparsity=self.track_sparsity,
            explosion_threshold=self.explosion_threshold,
        )
