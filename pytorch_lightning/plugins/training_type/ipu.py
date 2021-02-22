import json
import os
from typing import Any, Iterable, Optional, Union

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import _logger as log
from pytorch_lightning import LightningModule
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin
from pytorch_lightning.utilities import _POPTORCH_AVAILABLE
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _POPTORCH_AVAILABLE:
    import poptorch

    if not poptorch.ipuHardwareIsAvailable():
        raise MisconfigurationException("IPU Accelerator requires IPUs to run.")

# todo: No idea what's happening with grad accumulation, need to check since IPUs handle grad accum.
# todo: or even lr scheduling...


class LightningIPUModule(_LightningModuleWrapperBase):

    def __init__(self, pl_module: LightningModule, precision: int):
        super().__init__(pl_module)
        self.precision = precision

    def forward(self, *inputs, **kwargs):
        if self.precision == 16:
            inputs = self._move_float_tensors_to_half(inputs)

        return super().forward(*inputs, **kwargs)

    @staticmethod
    def batch_to(data):
        return data.half()

    def _move_float_tensors_to_half(self, batch: Any):
        batch = apply_to_collection(batch, (torch.FloatTensor, torch.cuda.FloatTensor), function=self.batch_to)
        return batch


class IPUPlugin(TrainingTypePlugin):

    def __init__(
        self,
        mixed_precision: bool,
        half: bool = False,
        device_iterations: int = 1,
        replication_factor: int = 1,
        autoround_num_ipus: bool = True,
        autoreport: bool = True,
        autoreport_dir: Optional[str] = None
    ):
        super().__init__()
        self.half = half
        self.mixed_precision = mixed_precision
        self.device_iterations = device_iterations
        self.replication_factor = replication_factor
        self.autoround_num_ipus = autoround_num_ipus
        self.autoreport = autoreport
        self.autoreport_dir = autoreport_dir

        if self.autoreport:
            options = {"autoReport.all": self.autoreport}
            if self.autoreport_dir:
                if not os.path.exists(self.autoreport_dir):
                    os.makedirs(self.autoreport_dir)
                options["autoReport.directory"] = self.autoreport_dir
            os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(options)

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

    def broadcast(self, obj: object, src: int = 0) -> object:
        return object

    @property
    def lightning_module(self) -> Optional[LightningModule]:
        return self.model.module if isinstance(self.model, LightningIPUModule) else self.model

    def pre_dispatch(self) -> None:
        if self.half:
            log.info('Using 16bit precision, converting model to FP16.')
            self.model = self.model.half()
        precision = 16 if self.half or self.mixed_precision else 32

        # Separate models are instantiated for different stages, but they share the same weights on host.
        # When validation/test models are run, they sync weights first.
        # Create model for training which will run training.

        optimizer = self.lightning_module.trainer.optimizers[0]
        self.model = poptorch.trainingModel(
            model=LightningIPUModule(self.lightning_module, precision),
            options=self._create_opts(is_train_model=True),
            optimizer=optimizer
        )

        # Create model for training which will run validation.
        self.validation_model = LightningIPUModule(self.lightning_module, precision)
        self.validation_model = poptorch.inferenceModel(
            model=self.validation_model,
            options=self._create_opts(is_train_model=False),
        )

    def _create_opts(self, is_train_model):
        opts = poptorch.Options()
        opts.deviceIterations(self.device_iterations)
        opts.replicationFactor(self.replication_factor)
        gradient_accumulation = self.lightning_module.trainer.accumulate_grad_batches if is_train_model else 1
        opts.Training.gradientAccumulation(gradient_accumulation)
        opts.autoRoundNumIPUs(self.autoround_num_ipus)
        return opts

    def process_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        dataloader = self._convert_to_poptorch_loader(
            dataloader=dataloader, opts=self._create_opts(is_train_model=self.lightning_module.training)
        )
        return dataloader

    def _convert_to_poptorch_loader(self, dataloader, opts):
        skip_keys = ['dataset_kind']
        if dataloader.batch_size:
            # re-create batch sampler in new poptorch loader
            skip_keys += ['batch_sampler']

        dl_args = {k: v for k, v in dataloader.__dict__.items() if not k.startswith('_') and k not in skip_keys}
        dl_args["options"] = opts
        multiprocessing_context = dataloader.multiprocessing_context
        dataloader = poptorch.DataLoader(**dl_args)
        dataloader.multiprocessing_context = multiprocessing_context
        return dataloader
