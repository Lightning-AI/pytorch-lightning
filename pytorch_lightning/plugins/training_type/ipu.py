import inspect
import json
import os
from typing import Any, Iterable, List, Optional, Union

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from pytorch_lightning import _logger as log
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities import _POPTORCH_AVAILABLE
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _POPTORCH_AVAILABLE:
    import poptorch

# todo: Check gradient accumulation to ensure this works, similar to DeepSpeed IPUs manage this.
# todo: Check lr scheduling to ensure that when the LR is changed, we update the optimizer state.


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


class IPUPlugin(ParallelPlugin):

    def __init__(
        self,
        half: bool = False,
        device_iterations: int = 1,
        autoround_num_ipus: bool = True,
        autoreport: bool = True,
        autoreport_dir: Optional[str] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
    ):
        super().__init__(parallel_devices, cluster_environment)
        self.half = half
        self.device_iterations = device_iterations
        self.autoround_num_ipus = autoround_num_ipus
        self.autoreport = autoreport
        self.autoreport_dir = autoreport_dir
        self.poptorch_models = {}

        if self.autoreport:
            options = {"autoReport.all": self.autoreport}
            if self.autoreport_dir:
                if not os.path.exists(self.autoreport_dir):
                    os.makedirs(self.autoreport_dir)
                options["autoReport.directory"] = self.autoreport_dir
            os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(options)

    def setup(self, model: Module) -> None:
        super().setup(model)
        if not poptorch.ipuHardwareIsAvailable():
            raise MisconfigurationException("IPU Accelerator requires IPUs to run.")

    @property
    def lightning_module(self) -> Optional[LightningModule]:
        return self.model.module if isinstance(self.model, LightningIPUModule) else self.model

    def pre_dispatch(self) -> None:
        if self.half:
            log.info('Using full 16bit precision, converting LightningModule weights to FP16.')
            self.model = self.model.half()
        precision = self.lightning_module.trainer.accelerator.precision_plugin.precision
        precision = 16 if self.half else precision

        model = LightningIPUModule(self.lightning_module, precision)
        self.model = model

        # Separate models are instantiated for different stages, but they share the same weights on host.
        # When validation/test models are run, they sync weights first.

        if self.lightning_module.trainer.training:
            # Create model for training which will run training.
            optimizer = self.lightning_module.trainer.optimizers[0]
            self.poptorch_models['train'] = poptorch.trainingModel(
                model=model, options=self._create_opts(training=True), optimizer=optimizer
            )
        for x in ('val', 'test', 'predict'):
            self.poptorch_models[x] = poptorch.inferenceModel(
                model=model,
                options=self._create_opts(training=False),
            )

    @property
    def replication_factor(self):
        return len(self.parallel_devices)

    def _create_opts(self, training):
        opts = poptorch.Options()
        opts.deviceIterations(self.device_iterations)
        opts.replicationFactor(self.replication_factor)
        gradient_accumulation = self.lightning_module.trainer.accumulate_grad_batches if training else 1
        opts.Training.gradientAccumulation(gradient_accumulation)
        opts.autoRoundNumIPUs(self.autoround_num_ipus)

        # todo (sean): unsure if this is necessary but to be safe.
        if os.environ.get("PL_GLOBAL_SEED"):
            opts.randomSeed(int(os.environ["PL_GLOBAL_SEED"]))
        return opts

    def on_reset_train_dataloader(self, dataloader) -> Union[Iterable, DataLoader]:
        return self.process_dataloader(dataloader)

    def on_reset_val_dataloader(self, dataloader) -> Union[Iterable, DataLoader]:
        return self.process_dataloader(dataloader)

    def on_reset_test_dataloader(self, dataloader) -> Union[Iterable, DataLoader]:
        return self.process_dataloader(dataloader)

    def on_reset_predict_dataloader(self, dataloader) -> Union[Iterable, DataLoader]:
        return self.process_dataloader(dataloader)

    def process_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        if isinstance(dataloader, CombinedLoader):
            dataloader.loaders = apply_to_collection(
                dataloader.loaders,
                DataLoader,
                self.process_dataloader,
            )
            return dataloader
        elif isinstance(dataloader, list):
            dataloader = apply_to_collection(dataloader, DataLoader, self.process_dataloader)
            return dataloader
        if not isinstance(dataloader, poptorch.DataLoader):
            dataloader = self._convert_to_poptorch_loader(
                dataloader=dataloader, opts=self._create_opts(training=self.lightning_module.training)
            )
        return dataloader

    def _convert_to_poptorch_loader(self, dataloader: Union[Iterable, DataLoader],
                                    opts: 'poptorch.Options') -> Union[Iterable, DataLoader]:
        skip_keys = ('sampler', 'batch_sampler', 'dataset_kind')

        attrs = {k: v for k, v in vars(dataloader).items() if not k.startswith("_")}

        params = set(inspect.signature(dataloader.__init__).parameters)
        contains_dataset = True

        if type(dataloader) is not DataLoader:
            contains_dataset = "dataset" in params
            params.update(inspect.signature(DataLoader.__init__).parameters)

        dl_args = {name: attrs[name] for name in params if name in attrs and name not in skip_keys}

        multiprocessing_context = dataloader.multiprocessing_context
        dl_args['multiprocessing_context'] = multiprocessing_context
        if not contains_dataset:
            dl_args.pop('dataset')

        dataloader = poptorch.DataLoader(**dl_args, options=opts)
        dataloader.multiprocessing_context = multiprocessing_context
        return dataloader

    def training_step(self, *args, **kwargs):
        args = self._prepare_input(args)
        return self.poptorch_models['train'](*args, **kwargs)

    def _prepare_input(self, args):
        # Ensure we replicate primitives values to have enough dimensions to split across devices
        accumulate_grad_batches = self.lightning_module.trainer.accumulate_grad_batches
        num_repeat = self.replication_factor * self.device_iterations * accumulate_grad_batches

        def to_tuple(x):
            return tuple(x)

        def to_tensor(x):
            return torch.tensor(x).unsqueeze(0).repeat(num_repeat)

        args = apply_to_collection(args, dtype=list, function=to_tuple)
        args = apply_to_collection(args, dtype=(int, float), function=to_tensor)
        return args

    def validation_step(self, *args, **kwargs):
        args = self._prepare_input(args)
        return self.poptorch_models['val'](*args, **kwargs)

    def test_step(self, *args, **kwargs):
        args = self._prepare_input(args)
        return self.poptorch_models['test'](*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        args = self._prepare_input(args)
        return self.poptorch_models['predict'](*args, **kwargs)

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
