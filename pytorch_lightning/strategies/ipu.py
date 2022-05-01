# Copyright The PyTorch Lightning team.
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
import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities import _IPU_AVAILABLE, _POPTORCH_AVAILABLE, rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.data import _get_dataloader_init_kwargs
from pytorch_lightning.utilities.enums import PrecisionType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import STEP_OUTPUT

if _POPTORCH_AVAILABLE:
    import poptorch
else:
    poptorch = None


class LightningIPUModule(_LightningModuleWrapperBase):
    def __init__(
        self, pl_module: Union["pl.LightningModule", _LightningPrecisionModuleWrapperBase], precision: Union[str, int]
    ) -> None:
        super().__init__(pl_module)
        self.precision = precision

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        if self.precision in (PrecisionType.MIXED, PrecisionType.HALF):
            inputs = self._move_float_tensors_to_half(inputs)

        return super().forward(*inputs, **kwargs)

    @staticmethod
    def batch_to(data: torch.Tensor) -> torch.Tensor:
        return data.half()

    def _move_float_tensors_to_half(self, batch: Any) -> Any:
        batch = apply_to_collection(batch, (torch.FloatTensor, torch.cuda.FloatTensor), function=self.batch_to)
        return batch


class IPUStrategy(ParallelStrategy):
    """Plugin for training on IPU devices."""

    strategy_name = "ipu_strategy"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        device_iterations: int = 1,
        autoreport: bool = False,
        autoreport_dir: Optional[str] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        training_opts: Optional["poptorch.Options"] = None,
        inference_opts: Optional["poptorch.Options"] = None,
    ) -> None:
        """
        Arguments:

            device_iterations: Number of iterations to run on device at once before returning to host.
                This can be used as an optimization to speed up training.
                https://docs.graphcore.ai/projects/poptorch-user-guide/en/0.1.67/batching.html
            autoreport: Enable auto-reporting for IPUs using PopVision
                https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/graph/graph.html
            autoreport_dir: Optional directory to store autoReport output.
            training_opts: Optional ``poptorch.Options`` to override the default created options for training.
            inference_opts: Optional ``poptorch.Options`` to override the default
                created options for validation/testing and predicting.
        """
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        if not _IPU_AVAILABLE:
            raise MisconfigurationException(
                "The IPU Accelerator requires IPU devices to run. "
                "Learn more or get started with IPUs at https://www.graphcore.ai/getstarted"
            )

        self.device_iterations = device_iterations
        self.autoreport = autoreport
        self.autoreport_dir = autoreport_dir
        self.poptorch_models = {}
        self._training_opts = training_opts
        self._inference_opts = inference_opts

        if self.autoreport:
            options = {"autoReport.all": self.autoreport}
            if self.autoreport_dir:
                self._fs = get_filesystem(str(self.autoreport_dir))
                self._fs.makedirs(self.autoreport_dir, exist_ok=True)
                options["autoReport.directory"] = self.autoreport_dir
            os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(options)

        self._update_dataloader_original: Optional[Callable] = None
        self._optimizer_zero_grad_original: Optional[Callable] = None

    def setup(self, trainer: "pl.Trainer") -> None:
        # set the `accumulate_grad_batches` property as early as possible
        self._handle_gradient_accumulation_steps()

        # patch the dataloader creation function with the custom `poptorch.DataLoader`.
        # this violates the intended control flow for the plugins, but since this is experimental, we have chosen
        # to use the simpler solution before adding abstractions to override the `DataLoader` class
        self._update_dataloader_original = pl.trainer.connectors.data_connector._update_dataloader
        pl.trainer.connectors.data_connector._update_dataloader = self._convert_to_poptorch_loader

        super().setup(trainer)

        # disable the `optimizer_zero_grad` function by setting it to `None`.
        # this is because the IPU zeros the gradients internally
        self._optimizer_zero_grad_original = self.lightning_module.optimizer_zero_grad
        self._disable_zero_grad()

        model = LightningIPUModule(self.lightning_module, self.precision_plugin.precision)
        self.model = model

        # reset the backup
        self.poptorch_models = {}

        # Separate models are instantiated for different stages, but they share the same weights on host.
        # When validation/test models are run, weights are synced first.
        trainer_fn = self.lightning_module.trainer.state.fn
        if trainer_fn in (TrainerFn.FITTING, TrainerFn.TUNING):
            # Create model for training and validation which will run on fit
            training_opts = self.training_opts
            inference_opts = self.inference_opts
            optimizer = self.lightning_module.trainer.optimizers[0]
            model = poptorch.trainingModel(model=model, options=training_opts, optimizer=optimizer)
            self.poptorch_models[RunningStage.TRAINING] = model

            if self.lightning_module.trainer.enable_validation:
                model = poptorch.inferenceModel(model=model, options=inference_opts)
                self.poptorch_models[RunningStage.VALIDATING] = model
        elif trainer_fn == TrainerFn.VALIDATING:
            model = poptorch.inferenceModel(model=model, options=self.inference_opts)
            self.poptorch_models[RunningStage.VALIDATING] = model
        elif trainer_fn == TrainerFn.TESTING:
            model = poptorch.inferenceModel(model=model, options=self.inference_opts)
            self.poptorch_models[RunningStage.TESTING] = model
        elif trainer_fn == TrainerFn.PREDICTING:
            model = poptorch.inferenceModel(model=model, options=self.inference_opts)
            self.poptorch_models[RunningStage.PREDICTING] = model

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        super().setup_optimizers(trainer)

        if len(self.optimizers) > 1:
            raise MisconfigurationException("IPUs currently only support one optimizer.")

    @property
    def replication_factor(self) -> int:
        if not self.lightning_module or not self.poptorch_models:
            # The plugin has been passed in by the user and has not been connected to the Trainer.
            # Check if the user has passed in custom poptorch.Options to infer number of IPUs being used.
            # In this scenario we prioritize the training options.
            if self._training_opts:
                return self._training_opts.replication_factor
            if self._inference_opts:
                return self._inference_opts.replication_factor

            return len(self.parallel_devices)

        stage = self.lightning_module.trainer.state.stage
        return self.poptorch_models[stage]._options.toDict()["replication_factor"]

    def _create_opts(self, training: bool) -> "poptorch.Options":
        opts = poptorch.Options()
        opts.deviceIterations(self.device_iterations)
        opts.replicationFactor(self.replication_factor)
        gradient_accumulation = self.lightning_module.trainer.accumulate_grad_batches if training else 1
        opts.Training.gradientAccumulation(gradient_accumulation)

        if os.environ.get("PL_GLOBAL_SEED"):
            opts.randomSeed(int(os.environ["PL_GLOBAL_SEED"]))
        return opts

    @property
    def training_opts(self) -> "poptorch.Options":
        if self._training_opts is None:
            self._training_opts = self._create_opts(training=True)
        return self._training_opts

    @property
    def inference_opts(self) -> "poptorch.Options":
        if self._inference_opts is None:
            self._inference_opts = self._create_opts(training=False)
        return self._inference_opts

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        return self.model.module if isinstance(self.model, LightningIPUModule) else self.model

    def _convert_to_poptorch_loader(
        self, dataloader: DataLoader, sampler, mode: Optional[RunningStage] = None
    ) -> "poptorch.DataLoader":
        if isinstance(dataloader, poptorch.DataLoader):
            # the user is returning the `poptorch.DataLoader` directly, don't change anything.
            return dataloader

        dl_kwargs = _get_dataloader_init_kwargs(dataloader, sampler)
        opts = self.training_opts if mode == RunningStage.TRAINING else self.inference_opts
        dataloader = poptorch.DataLoader(opts, **dl_kwargs)
        return dataloader

    def _handle_gradient_accumulation_steps(self) -> None:
        """Override the trainer.accumulation_scheduler to act as ``accumulate_grad_batches=1`` if gradient
        accumulation has been set.

        ``optimizer_step`` will be called on every batch, and the IPU will handle grad accumulation internally.
        """
        accumulation_scheduler = self.lightning_module.trainer.accumulation_scheduler

        if accumulation_scheduler.epochs != [0]:
            raise MisconfigurationException(
                "IPUs currently does not support different `accumulate_grad_batches` at different epochs."
            )

        # TODO(@tchaton): Add support for accumulate_grad_batches being a dictionary
        accumulation_scheduler.scheduling.update({0: 1})

    @property
    def _n_replicate(self):
        opts = self.training_opts if self.lightning_module.training else self.inference_opts
        accumulate_grad_batches = opts.Training.gradient_accumulation
        device_iterations = opts.device_iterations
        replication_factor = opts.replication_factor
        return replication_factor * device_iterations * accumulate_grad_batches

    def _prepare_input(self, args: Any):
        def to_tuple(x):
            return tuple(x)

        def to_tensor(x):
            return torch.tensor(x).unsqueeze(0).repeat(self._n_replicate)

        args = apply_to_collection(args, dtype=list, function=to_tuple)
        args = apply_to_collection(args, dtype=(int, float), function=to_tensor)
        return args

    def _disable_zero_grad(self) -> None:
        lightning_module = self.lightning_module
        if is_overridden("optimizer_zero_grad", lightning_module):
            assert lightning_module is not None  # `is_overridden` returns False otherwise
            rank_zero_warn(
                "You have overridden the `LightningModule.optimizer_zero_grad` hook but it will be ignored since"
                " IPUs handle the zeroing of gradients internally."
            )
        lightning_module.optimizer_zero_grad = None  # type: ignore[assignment]

    def _step(self, stage: RunningStage, *args: Any, **kwargs: Any):
        args = self._prepare_input(args)
        poptorch_model = self.poptorch_models[stage]
        self.lightning_module._running_torchscript = True
        out = poptorch_model(*args, **kwargs)
        self.lightning_module._running_torchscript = False
        return out

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.train_step_context():
            return self._step(RunningStage.TRAINING, *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.val_step_context():
            return self._step(RunningStage.VALIDATING, *args, **kwargs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.test_step_context():
            return self._step(RunningStage.TESTING, *args, **kwargs)

    def predict_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.predict_step_context():
            return self._step(RunningStage.PREDICTING, *args, **kwargs)

    def teardown(self) -> None:
        super().teardown()
        if self._update_dataloader_original is not None:
            # undo dataloader patching
            pl.trainer.connectors.data_connector._update_dataloader = self._update_dataloader_original

        if self._optimizer_zero_grad_original is not None:
            # re-enable `optimizer_zero_grad`
            self.lightning_module.optimizer_zero_grad = self._optimizer_zero_grad_original

        for model in self.poptorch_models.values():
            model.destroy()

    def _compiled(self, model: Any):
        # Required to ensure we only attach compiled models, as they are compiled lazily.
        return model._executable is not None

    def _detach_models(self):
        """Detaches all stage specific models from IPU devices."""
        for k, model in self.poptorch_models.items():
            if self._compiled(model) and model.isAttachedToDevice():
                model.detachFromDevice()

    def _load_model(self, stage: str):
        """Loads the stage specific accelerator model onto device if compiled and not attached to IPU devices.

        Args:
            stage: The stage to load
        """
        self._detach_models()
        model = self.poptorch_models[stage]
        if self._compiled(model) and not model.isAttachedToDevice():
            model.attachToDevice()

    def on_train_start(self):
        self._load_model(RunningStage.TRAINING)

    def on_validation_start(self):
        self._load_model(RunningStage.VALIDATING)

    def on_test_start(self):
        self._load_model(RunningStage.TESTING)

    def on_predict_start(self):
        self._load_model(RunningStage.PREDICTING)

    def on_train_end(self):
        self._detach_models()

    def on_validation_end(self):
        self._detach_models()

    def on_test_end(self):
        self._detach_models()

    def on_predict_end(self):
        self._detach_models()

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Updates optimizer stats if LR scheduler modified the optimizer state
        optimizer = self.optimizers[0]
        self.poptorch_models[RunningStage.TRAINING].setOptimizer(optimizer)

    @property
    def root_device(self) -> torch.device:
        pass

    def model_to_device(self) -> None:
        pass

    @property
    def is_global_zero(self) -> bool:
        return True

    def reduce(self, tensor: Union[torch.Tensor, Any], *args: Any, **kwargs: Any) -> Union[torch.Tensor, Any]:
        return tensor

    def barrier(self, name: Optional[str] = None) -> None:
        pass

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        return tensor

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
