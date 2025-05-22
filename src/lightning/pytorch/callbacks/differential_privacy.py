__all__ = ["DifferentialPrivacy"]

import typing as ty
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

try:
    from opacus import GradSampleModule
    from opacus.accountants import RDPAccountant
    from opacus.accountants.utils import get_noise_multiplier
    from opacus.data_loader import DPDataLoader
    from opacus.layers.dp_rnn import DPGRUCell
    from opacus.optimizers import DPOptimizer as OpacusDPOptimizer
except ImportError as ex:
    raise ImportError("Opacus is not installed. Please install it with `pip install opacus`.") from ex

import pytorch_lightning as pl


def replace_grucell(module: torch.nn.Module) -> None:
    """Replaces GRUCell modules with DP-counterparts."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GRUCell) and not isinstance(child, DPGRUCell):
            replacement = copy_gru(child)
            setattr(module, name, replacement)
    for name, child in module.named_children():
        replace_grucell(child)


def copy_gru(grucell: torch.nn.GRUCell) -> DPGRUCell:
    """Creates a DP-GRUCell from a non-DP one."""
    input_size: int = grucell.input_size
    hidden_size: int = grucell.hidden_size
    bias: bool = grucell.bias
    dpgrucell = DPGRUCell(input_size, hidden_size, bias)
    for name, param in grucell.named_parameters():
        if "ih" in name:
            _set_layer_param(dpgrucell, name, param, "ih")
        elif "hh" in name:
            _set_layer_param(dpgrucell, name, param, "hh")
        else:
            raise AttributeError(f"Unknown parameter {name}")
    return dpgrucell


def _set_layer_param(
    dpgrucell: DPGRUCell,
    name: str,
    param: torch.Tensor,
    layer_name: str,
) -> None:
    """Helper"""
    layer = getattr(dpgrucell, layer_name)
    if "weight" in name:
        layer.weight = torch.nn.Parameter(deepcopy(param))
    elif "bias" in name:
        layer.bias = torch.nn.Parameter(deepcopy(param))
    else:
        raise AttributeError(f"Unknown parameter {name}")
    setattr(dpgrucell, layer_name, layer)


def params(
    optimizer: torch.optim.Optimizer,
    accepted_names: list[str] = None,
) -> list[torch.nn.Parameter]:
    """
    Return all parameters controlled by the optimizer
    Args:
        accepted_names (list[str]):
            List of parameter group names you want to apply DP to.
            This allows you to choose to apply DP only to specific parameter groups.
            Of course, this will work only if the optimizer has named parameter groups.
            If it doesn't, then this argument will be ignored and DP will be applied to all parameter groups.
    Returns:
        (list[torch.nn.Parameter]): Flat list of parameters from all `param_groups`
    """
    # lower case
    if accepted_names is not None:
        accepted_names = [name.lower() for name in accepted_names]
    # unwrap parameters from the param_groups into a flat list
    ret = []
    for param_group in optimizer.param_groups:
        if accepted_names is not None and "name" in param_group:
            name: str = param_group["name"].lower()
            if name.lower() in accepted_names:
                ret += [p for p in param_group["params"] if p.requires_grad]
        else:
            ret += [p for p in param_group["params"] if p.requires_grad]
    return ret


class DPOptimizer(OpacusDPOptimizer):
    """Brainiac-2's DP-Optimizer"""

    def __init__(
        self,
        *args: ty.Any,
        param_group_names: list[str] = None,
        **kwargs: ty.Any,
    ) -> None:
        """Constructor."""
        self.param_group_names = param_group_names
        super().__init__(*args, **kwargs)

    @property
    def params(self) -> list[torch.nn.Parameter]:
        """
        Returns a flat list of ``nn.Parameter`` managed by the optimizer
        """
        return params(self, self.param_group_names)


class DifferentialPrivacy(pl.callbacks.EarlyStopping):
    """Enables differential privacy using Opacus.
    Converts optimizers to instances of the :class:`~opacus.optimizers.DPOptimizer` class.
    This callback inherits from `EarlyStopping`, thus it is also able to stop the
    training when enough privacy budget has been spent.
    Please beware that Opacus does not support multi-optimizer training.
    For more info, check the following links:
    * https://opacus.ai/tutorials/
    * https://blog.openmined.org/differentially-private-deep-learning-using-opacus-in-20-lines-of-code/
    """

    def __init__(
        self,
        budget: float = 1.0,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        delta: float = None,
        use_target_values: bool = False,
        idx: ty.Sequence[int] = None,
        log_spent_budget_as: str = "DP/spent-budget",
        param_group_names: list[str] = None,
        private_dataloader: bool = False,
        default_alphas: ty.Sequence[ty.Union[float, int]] = None,
        **gsm_kwargs: ty.Any,
    ) -> None:
        """Enables differential privacy using Opacus.
        Converts optimizers to instances of the :class:`~opacus.optimizers.DPOptimizer` class.
        This callback inherits from `EarlyStopping`,
        thus it is also able to stop the training when enough privacy budget has been spent.
        Args:
            budget (float, optional): Defaults to 1.0.
                Maximun privacy budget to spend.
            noise_multiplier (float, optional): Defaults to 1.0.
                Noise multiplier.
            max_grad_norm (float, optional): Defaults to 1.0.
                Max grad norm used for gradient clipping.
            delta (float, optional): Defaults to None.
                The target δ of the (ϵ,δ)-differential privacy guarantee.
                Generally, it should be set to be less than the inverse of the size of the training dataset.
                If `None`, this will be set to the inverse of the size of the training dataset `N`: `1/N`.
            use_target_values (bool, optional):
                Whether to call `privacy_engine.make_private_with_epsilon()` or `privacy_engine.make_private`.
                If `True`, the value of `noise_multiplier` will be calibrated automatically so that the desired privacy
                budget will be reached only at the end of the training.
            idx (ty.Sequence[int]):
                List of optimizer ID's to make private. Useful when a model may have more than one optimizer.
                By default, all optimizers are made private.
            log_spent_budget_as (str, optional):
                How to log and expose the spent budget value.
                Although this callback already allows you to stop the training when
                enough privacy budget has been spent (see argument `stop_on_budget`),
                this keyword argument can be used in combination with an `EarlyStopping`
                callback, so that the latter may use this value to stop the training when enough budget has been spent.
            param_group_names (list[str]):
                List of parameter group names you want to apply DP to. This allows you
                to choose to apply DP only to specific parameter groups. Of course, this
                will work only if the optimizer has named parameter groups. If it
                doesn't, then this argument will be ignored and DP will be applied to
                all parameter groups.
            private_dataloader (bool, optional):
                Whether to make the dataloader private. Defaults to False.
            **gsm_kwargs:
                Input arguments for the :class:`~opacus.GradSampleModule` class.
        """
        # inputs
        self.budget = budget
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.use_target_values = use_target_values
        self.log_spent_budget_as = log_spent_budget_as
        self.param_group_names = param_group_names
        self.private_dataloader = private_dataloader
        self.gsm_kwargs = gsm_kwargs
        if default_alphas is None:
            self.default_alphas = RDPAccountant.DEFAULT_ALPHAS + list(range(64, 150))
        else:
            self.default_alphas = default_alphas
        # init early stopping callback
        super().__init__(
            monitor=self.log_spent_budget_as,
            mode="max",
            stopping_threshold=self.budget,
            check_on_train_epoch_end=True,
            # we do not want to stop if spent budget does not increase. this may even be desirable
            min_delta=0,
            patience=1000000,
        )
        # attributes
        self.epsilon: float = 0.0
        self.best_alpha: float = 0.0
        self.accountant = RDPAccountant()
        self.idx = idx  # optims to privatize

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str = None,
    ) -> None:
        """Call the GradSampleModule() wrapper to add attributes to pl_module."""
        if stage == "fit":
            replace_grucell(pl_module)
            try:
                pl_module = GradSampleModule(pl_module, **self.gsm_kwargs)
            except ImportError as ex:
                raise ImportError(f"{ex}. This may be due to a mismatch between Opacus and PyTorch version.") from ex

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called when the training epoch begins. Use this to make optimizers private."""
        # idx
        if self.idx is None:
            self.idx = range(len(trainer.optimizers))

        # Replace current dataloaders with private counterparts
        expected_batch_size = 1
        cl = trainer.fit_loop._combined_loader
        if cl is not None:
            dp_dls: list[DPDataLoader] = []
            for i, dl in enumerate(cl.flattened):
                if isinstance(dl, DataLoader):
                    sample_rate: float = 1 / len(dl)
                    dataset_size: int = len(dl.dataset)  # type: ignore
                    expected_batch_size = int(dataset_size * sample_rate)
                    if self.private_dataloader:
                        dp_dl = DPDataLoader.from_data_loader(dl, distributed=False)
                        dp_dls.append(dp_dl)
            # it also allows you to easily replace the dataloaders
            if self.private_dataloader:
                cl.flattened = dp_dls

        # Delta
        if self.delta is None:
            self.delta = 1 / dataset_size

        # Make optimizers private
        optimizers: list[Optimizer] = []
        dp_optimizer: ty.Union[Optimizer, DPOptimizer]
        for i, optimizer in enumerate(trainer.optimizers):
            if not isinstance(optimizer, DPOptimizer) and i in self.idx:
                if self.use_target_values:
                    self.noise_multiplier = get_noise_multiplier(
                        target_epsilon=self.budget / 2,
                        target_delta=self.delta,
                        sample_rate=sample_rate,
                        epochs=trainer.max_epochs,
                        accountant="rdp",
                    )
                dp_optimizer = DPOptimizer(
                    optimizer=optimizer,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                    expected_batch_size=expected_batch_size,
                    param_group_names=self.param_group_names,
                )
                dp_optimizer.attach_step_hook(self.accountant.get_optimizer_hook_fn(sample_rate=sample_rate))
            else:
                dp_optimizer = optimizer
            optimizers.append(dp_optimizer)
        # Replace optimizers
        trainer.optimizers = optimizers

    def on_train_batch_end(  # pylint: disable=unused-argument # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: ty.Any,
        batch: ty.Any,
        batch_idx: int,
        *args: ty.Any,
    ) -> None:
        """Called after the batched has been digested. Use this to understand whether to stop or not."""
        self._log_and_stop_criterion(trainer, pl_module)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run at the end of the training epoch."""

    def get_privacy_spent(self) -> tuple[float, float]:
        """Estimate spent budget."""
        # get privacy budget spent so far
        epsilon, best_alpha = self.accountant.get_privacy_spent(
            delta=self.delta,
            alphas=self.default_alphas,
        )
        return float(epsilon), float(best_alpha)

    def _log_and_stop_criterion(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Logging privacy spent: (epsilon, delta) and stopping if necessary."""
        self.epsilon, self.best_alpha = self.get_privacy_spent()
        pl_module.log(
            self.log_spent_budget_as,
            self.epsilon,
            on_epoch=True,
            prog_bar=True,
        )
        if self.epsilon > self.budget:
            trainer.should_stop = True
