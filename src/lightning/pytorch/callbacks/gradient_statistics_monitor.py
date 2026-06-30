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

from lightning.fabric.plugins.precision.amp import _optimizer_handles_unscaling
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
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
        - Per-step metrics are logged under ``train/`` every ``log_every_n_steps`` global
          steps (e.g. ``train/grad_norm``, ``train/grad_mean``).
        - Per-epoch metrics are logged at the end of every epoch under ``train_epoch/``
          (e.g. ``train_epoch/grad_norm``), aggregated over all optimizer steps regardless
          of ``log_every_n_steps``.
        - Logging is performed only on the global rank (for distributed training safety).
        - Uses Lightning's ``log_dict`` for compatibility with all supported loggers.
        - The epoch accumulator and step counter are saved in checkpoints via ``state_dict`` /
          ``load_state_dict``, so epoch aggregates remain correct after a mid-epoch resume.

    Subclassing:
        Override any of the following to customise what is computed or logged:

        - ``step_prefix`` — property that controls the per-step metric namespace (default ``"train/"``)
        - ``epoch_prefix`` — property that controls the per-epoch metric namespace (default ``"train_epoch/"``)
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
        - When using AMP (``precision="16-mixed"``), ``GradScaler`` skips the optimizer update
          if non-finite gradients are detected, but this callback has already logged by then.
          Stats logged on such steps reflect gradients that never produced a parameter update.

    """

    def __init__(
        self,
        track_batches: bool = True,
        track_epochs: bool = True,
        per_layer: bool = False,
        track_sparsity: bool = True,
        explosion_threshold: float | None = 1e4,
    ):
        super().__init__()
        if not track_batches and not track_epochs:
            raise MisconfigurationException("GradientStatsMonitor must track at least one of batches or epochs.")
        self.track_batches = track_batches
        self.track_epochs = track_epochs
        self.per_layer = per_layer
        self.track_sparsity = track_sparsity
        self.explosion_threshold = explosion_threshold
        self._train_stats: dict[str, Any] = self.init_epoch_stats()
        self._last_logged_step: int = -1

    # -------------------------
    # Internal helpers
    # -------------------------

    def _on_grad_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Any,
        should_log: bool,
    ) -> None:
        layer_grads = self._collect_grads(trainer, pl_module, optimizer)
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
            metrics = self.compute_batch_stats(layer_grads)
            if threshold is not None:
                norm = metrics.get(f"{self.step_prefix}grad_norm")
                if norm is not None and norm > threshold:
                    self._warn_explosion(norm)
            self._log_scalars(trainer, pl_module, metrics)
        else:
            # Explosion check only — compute just the norm, not the full stat set.
            norm = sum(g.norm(2).item() ** 2 for g in layer_grads.values()) ** 0.5
            if threshold is not None and norm > threshold:
                self._warn_explosion(norm)

    def _collect_grads(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Any
    ) -> dict[str, torch.Tensor] | None:
        """Collect per-layer gradients, unscaling AMP-scaled values when necessary.

        When AMP is active and the optimizer handles its own unscaling (e.g. fused Adam),
        Lightning skips ``scaler.unscale_()`` before ``on_before_optimizer_step``, so
        ``param.grad`` is still multiplied by the scaler's scale factor.  In that case
        ``inv_scale`` cancels it out; otherwise it is ``1.0`` and has no effect.

        Returns ``{param_name: flat_grad_tensor}`` or ``None`` if no gradients exist.

        """
        from lightning.fabric.plugins.precision.amp import MixedPrecision

        pp = trainer.precision_plugin
        if isinstance(pp, MixedPrecision) and pp.scaler is not None and _optimizer_handles_unscaling(optimizer):
            inv_scale = 1.0 / pp.scaler.get_scale()
        else:
            inv_scale = 1.0

        layer_grads = {
            name: (param.grad.detach() * inv_scale).view(-1)
            for name, param in pl_module.named_parameters()
            if param.grad is not None
        }
        return layer_grads or None

    def _log_scalars(self, trainer: pl.Trainer, pl_module: pl.LightningModule, metrics: dict[str, float]) -> None:
        pl_module.log_dict(metrics)

    def _warn_explosion(self, norm: float) -> None:
        rank_zero_warn(f"Gradient norm is very high ({norm:.2f}). Possible exploding gradients.")

    # -------------------------
    # Metric prefixes  ->  override to change where metrics appear in the logger
    # -------------------------

    @property
    def step_prefix(self) -> str:
        """Metric prefix used for per-step stats (e.g. ``"train/"`` → ``train/grad_norm``)."""
        return "train/"

    @property
    def epoch_prefix(self) -> str:
        """Metric prefix used for per-epoch stats (e.g. ``"train_epoch/"`` → ``train_epoch/grad_norm``)."""
        return "train_epoch/"

    # -------------------------
    # Per-step stats  ->  override to control what is computed from each step's gradients
    # -------------------------

    def compute_batch_stats(self, layer_grads: dict[str, torch.Tensor]) -> dict[str, float]:
        """Compute and return the metric dict logged after each optimizer step.

        The returned dict is passed directly to ``pl_module.log_dict``.
        Override to add, remove, or rename metrics.

        """
        p = self.step_prefix
        names = list(layer_grads.keys())
        grads = list(layer_grads.values())
        total_count = sum(g.numel() for g in grads)

        layers_norm_1 = torch._foreach_norm(grads, 1)
        layers_norm_2 = torch._foreach_norm(grads, 2)

        metrics: dict[str, float] = {}
        if self.per_layer:
            for name, sq_val in zip(names, layers_norm_2):
                metrics[f"{p}grad_norm/{name.replace('.', '/')}"] = sq_val**0.5

        global_mean = sum(layers_norm_1) / total_count
        global_norm_2 = sum(layers_norm_2)
        metrics[f"{p}grad_norm"] = global_norm_2**0.5
        metrics[f"{p}grad_mean"] = global_mean
        metrics[f"{p}grad_std"] = global_norm_2 / total_count - global_mean**2
        if self.track_sparsity:
            sparse = [
                torch.where(g.abs() < _EPS, torch.tensor(1.0, device=g.device), torch.tensor(0.0, device=g.device))
                for g in grads
            ]
            metrics[f"{p}grad_sparsity"] = sum(torch._foreach_norm(sparse, 1)) / total_count
        return metrics

    # -------------------------
    # Per-epoch stats  ->  override to control accumulation and final epoch metrics
    # -------------------------

    def init_epoch_stats(self) -> dict[str, Any]:
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
        names = list(layer_grads.keys())
        grads = list(layer_grads.values())
        num_layers = len(grads)

        sq_sums: list[torch.Tensor] = [g.pow(2).sum() for g in grads]
        to_sync: list[torch.Tensor] = sq_sums + [g.sum() for g in grads]
        if self.track_sparsity:
            to_sync += [(g.abs() < _EPS).sum() for g in grads]

        vals = torch.stack(to_sync).tolist()  # one D2H sync

        sq_sum_vals = vals[:num_layers]
        sum_vals = vals[num_layers : 2 * num_layers]

        state["norm_sum"] += sum(sq_sum_vals) ** 0.5
        state["grad_count"] += sum(g.numel() for g in grads)
        state["grad_sum"] += sum(sum_vals)
        state["grad_sq_sum"] += sum(sq_sum_vals)
        if self.track_sparsity:
            state["near_zero_count"] += int(sum(vals[2 * num_layers :]))
        if self.per_layer:
            for name, sq_val in zip(names, sq_sum_vals):
                key = f"grad_norm/{name.replace('.', '/')}"
                state["layer_norm_sums"][key] = state["layer_norm_sums"].get(key, 0.0) + sq_val**0.5
        state["steps"] += 1

    def compute_epoch_stats(self, state: dict[str, Any]) -> dict[str, float] | None:
        """Compute and return the metric dict logged at the end of each epoch.

        Args:
            state: the accumulator produced by ``init_epoch_stats`` and updated
                by ``update_epoch_stats``.

        Returns ``None`` if no steps were recorded (empty dataloader).
        The returned dict is passed directly to ``pl_module.log_dict``.
        Override to add, remove, or rename metrics, or to derive additional
        values from extra state added in ``init_epoch_stats`` / ``update_epoch_stats``.

        Note:
            ``{epoch_prefix}grad_norm`` is the **mean of per-step global norms**
            (i.e. ``mean(‖g_t‖₂)`` over optimizer steps *t*), not the true L2 norm of all
            gradients accumulated over the epoch.  The same applies to per-layer norm averages.

        """
        if state["steps"] == 0:
            return None
        p = self.epoch_prefix
        metrics: dict[str, float] = {f"{p}grad_norm": state["norm_sum"] / state["steps"]}
        if state["grad_count"] > 0:
            mean = state["grad_sum"] / state["grad_count"]
            variance = state["grad_sq_sum"] / state["grad_count"] - mean**2
            metrics[f"{p}grad_mean"] = mean
            metrics[f"{p}grad_std"] = max(variance, 0.0) ** 0.5
        if self.track_sparsity and state["grad_count"] > 0:
            metrics[f"{p}grad_sparsity"] = state["near_zero_count"] / state["grad_count"]
        if self.per_layer:
            for key, norm_sum in state["layer_norm_sums"].items():
                metrics[f"{p}{key}"] = norm_sum / state["steps"]
        return metrics

    # -------------------------
    # Hooks
    # -------------------------

    @override
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy
        from lightning.pytorch.strategies.fsdp import FSDPStrategy
        from lightning.pytorch.strategies.model_parallel import ModelParallelStrategy

        if isinstance(trainer.strategy, (FSDPStrategy, DeepSpeedStrategy, ModelParallelStrategy)):
            raise MisconfigurationException(
                f"{type(trainer.strategy).__name__} is not supported by GradientStatsMonitor. "
                "Support for sharded strategies is planned for a future release. "
                "GradientStatsMonitor works correctly with single-device and DDP training."
            )

    @override
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._train_stats = self.init_epoch_stats()

    @override
    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Any) -> None:
        # Guard against multiple calls per step when there are multiple optimizers.
        if trainer.global_step == self._last_logged_step:
            return
        self._last_logged_step = trainer.global_step
        should_log = self.track_batches and trainer.global_step % trainer.log_every_n_steps == 0
        self._on_grad_step(trainer, pl_module, optimizer, should_log=should_log)

    @override
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.track_epochs:
            return
        metrics = self.compute_epoch_stats(self._train_stats)
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
            track_batches=self.track_batches,
            track_epochs=self.track_epochs,
            per_layer=self.per_layer,
            track_sparsity=self.track_sparsity,
            explosion_threshold=self.explosion_threshold,
        )
