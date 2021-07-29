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
import inspect
import json
import os
from typing import Any, Iterable, List, Optional, Union

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities import _POPTORCH_AVAILABLE
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _POPTORCH_AVAILABLE:
    import poptorch


class LightningIPUModule(_LightningModuleWrapperBase):
    def __init__(self, pl_module: "pl.LightningModule", precision: Union[str, int]):
        super().__init__(pl_module)
        self.precision = precision

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        if self.precision in ("mixed", 16):
            inputs = self._move_float_tensors_to_half(inputs)

        return super().forward(*inputs, **kwargs)

    @staticmethod
    def batch_to(data: torch.Tensor) -> torch.Tensor:
        return data.half()

    def _move_float_tensors_to_half(self, batch: Any) -> Any:
        batch = apply_to_collection(batch, (torch.FloatTensor, torch.cuda.FloatTensor), function=self.batch_to)
        return batch


class IPUPlugin(ParallelPlugin):
    """
    Plugin for training on IPU devices.
    """

    def __init__(
        self,
        device_iterations: int = 1,
        autoreport: bool = False,
        autoreport_dir: Optional[str] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
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
        super().__init__(parallel_devices, cluster_environment)
        if not _POPTORCH_AVAILABLE or not poptorch.ipuHardwareIsAvailable():
            raise MisconfigurationException(
                "The IPU Accelerator requires IPU devices to run. "
                "Learn more or get started with IPUs at https://www.graphcore.ai/getstarted"
            )

        self.device_iterations = device_iterations
        self.autoreport = autoreport
        self.autoreport_dir = autoreport_dir
        self.poptorch_models = {}
        self._original_accumulate_grad_batches = None
        self._training_opts = training_opts
        self._inference_opts = inference_opts

        if self.autoreport:
            options = {"autoReport.all": self.autoreport}
            if self.autoreport_dir:
                self._fs = get_filesystem(str(self.autoreport_dir))
                self._fs.makedirs(self.autoreport_dir, exist_ok=True)
                options["autoReport.directory"] = self.autoreport_dir
            os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(options)

    def pre_dispatch(self) -> None:
        precision = self.lightning_module.trainer.precision
        model = LightningIPUModule(self.lightning_module, precision)
        self.model = model

        # Separate models are instantiated for different stages, but they share the same weights on host.
        # When validation/test models are run, weights are synced first.

        if self.lightning_module.trainer.state.stage is RunningStage.TRAINING:
            # Create model for training which will run training.
            optimizer = self.lightning_module.trainer.optimizers[0]
            model = poptorch.trainingModel(model=model, options=self.training_opts, optimizer=optimizer)
            self.poptorch_models[RunningStage.TRAINING] = model
        for x in (RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING):
            model = poptorch.inferenceModel(model=model, options=self.inference_opts)
            self.poptorch_models[x] = model
        self._handle_gradient_accumulation_steps()

    @property
    def replication_factor(self) -> int:
        if not self.lightning_module:
            # The plugin has been passed in by the user and has not been connected to the Trainer.
            # Check if the user has passed in custom poptorch.Options to infer number of IPUs being used.
            # In this scenario we prioritize the training options.
            if self._training_opts:
                return self._training_opts.replication_factor
            if self._inference_opts:
                return self._inference_opts.replication_factor
        return len(self.parallel_devices)

    def _create_opts(self, training: bool) -> "poptorch.Options":
        opts = poptorch.Options()
        opts.deviceIterations(self.device_iterations)
        opts.replicationFactor(self.replication_factor)
        gradient_accumulation = self.accumulate_grad_batches if training else 1
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

    def on_reset_train_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        return self._process_dataloader(dataloader, is_training=True)

    def on_reset_val_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        return self._process_dataloader(dataloader, is_training=False)

    def on_reset_test_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        return self._process_dataloader(dataloader, is_training=False)

    def on_reset_predict_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        return self._process_dataloader(dataloader, is_training=False)

    def _process_dataloader(
        self, dataloader: Union[Iterable, DataLoader], is_training: bool
    ) -> Union[Iterable, DataLoader]:
        if isinstance(dataloader, CombinedLoader):
            dataloader.loaders = apply_to_collection(
                dataloader.loaders, DataLoader, self._process_dataloader, is_training
            )
            return dataloader
        if isinstance(dataloader, list):
            dataloader = apply_to_collection(dataloader, DataLoader, self._process_dataloader, is_training)
            return dataloader
        if not isinstance(dataloader, poptorch.DataLoader):
            opts = self.training_opts if is_training else self.inference_opts
            dataloader = self._convert_to_poptorch_loader(dataloader=dataloader, opts=opts)
        return dataloader

    def _convert_to_poptorch_loader(
        self, dataloader: Union[Iterable, DataLoader], opts: "poptorch.Options"
    ) -> Union[Iterable, DataLoader]:
        skip_keys = ("sampler", "batch_sampler", "dataset_kind")

        attrs = {k: v for k, v in vars(dataloader).items() if not k.startswith("_")}

        params = set(inspect.signature(dataloader.__init__).parameters)
        contains_dataset = True

        if type(dataloader) is not DataLoader:
            contains_dataset = "dataset" in params
            params.update(inspect.signature(DataLoader.__init__).parameters)

        dl_args = {name: attrs[name] for name in params if name in attrs and name not in skip_keys}

        multiprocessing_context = dataloader.multiprocessing_context
        dl_args["multiprocessing_context"] = multiprocessing_context
        if not contains_dataset:
            dl_args.pop("dataset")
        # Override to drop last uneven batch, as IPUs does not support uneven inputs.
        dl_args["drop_last"] = True

        dataloader = poptorch.DataLoader(**dl_args, options=opts)
        dataloader.multiprocessing_context = multiprocessing_context
        return dataloader

    @property
    def accumulate_grad_batches(self) -> int:
        """
        Tracks lazily the set accumulate_grad_batches in the trainer.
        The IPUPlugin replaces the original accumulate_grad_batches.
        """
        if self._original_accumulate_grad_batches is None:
            self._original_accumulate_grad_batches = self.lightning_module.trainer.accumulate_grad_batches
            if not isinstance(self._original_accumulate_grad_batches, int):
                raise MisconfigurationException(
                    "IPUs currently only support accumulate_grad_batches being an integer value. "
                    f"Received {self.accumulate_grad_batches}"
                )
        return self._original_accumulate_grad_batches

    def _handle_gradient_accumulation_steps(self):
        """
        This functions overrides the trainer.accumulation_scheduler to generate
        ``accumulate_grad_batches=1``.
        Therefore, ``optimizer_step`` will be called on every batch, and the IPU will handle grad accumulation.
        """
        if self.accumulate_grad_batches > 1:
            self.lightning_module.trainer.accumulation_scheduler = GradientAccumulationScheduler({0: 1})

    def update_global_step(self, total_batch_idx: int, current_global_step: int) -> int:
        if self.accumulate_grad_batches > 1:
            if total_batch_idx % self.accumulate_grad_batches == 0:
                current_global_step += 1
            return current_global_step
        return super().update_global_step(total_batch_idx, current_global_step)

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

    def training_step(self, *args, **kwargs):
        args = self._prepare_input(args)
        return self.poptorch_models[RunningStage.TRAINING](*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        args = self._prepare_input(args)
        return self.poptorch_models[RunningStage.VALIDATING](*args, **kwargs)

    def test_step(self, *args, **kwargs):
        args = self._prepare_input(args)
        return self.poptorch_models[RunningStage.TESTING](*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        args = self._prepare_input(args)
        return self.poptorch_models[RunningStage.PREDICTING](*args, **kwargs)

    def teardown(self) -> None:
        for model in self.poptorch_models.values():
            model.destroy()

    def _compiled(self, model: Any):
        # Required to ensure we only attach compiled models, as they are compiled lazily.
        return model._executable is not None

    def _detach_models(self):
        """
        Detaches all stage specific models from IPU devices.
        """
        for k, model in self.poptorch_models.items():
            if self._compiled(model) and model.isAttachedToDevice():
                model.detachFromDevice()

    def _load_model(self, stage: str):
        """
        Loads the stage specific accelerator model onto device if compiled and not attached to IPU devices.
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

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # Updates optimizer stats if LR scheduler modified the optimizer state
        optimizer = self.lightning_module.trainer.optimizers[0]
        self.poptorch_models[RunningStage.TRAINING].setOptimizer(optimizer)

    @property
    def on_gpu(self) -> bool:
        return False

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
