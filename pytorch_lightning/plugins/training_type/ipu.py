import inspect
import json
import os
from typing import Any, Iterable, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks import GradientAccumulationScheduler
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


class LightningIPUModule(_LightningModuleWrapperBase):

    def __init__(self, pl_module: LightningModule, precision: Union[str, int]):
        super().__init__(pl_module)
        self.precision = precision

    def forward(self, *inputs, **kwargs):
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
        autoround_num_ipus: bool = True,
        autoreport: bool = True,
        autoreport_dir: Optional[str] = None,
        convert_model_to_half: bool = False,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
    ) -> None:
        """
        Arguments:

            device_iterations: Number of iterations to run on device at once before returning to host.
                This can be used as an optimization to speed up training.
                https://docs.graphcore.ai/projects/poptorch-user-guide/en/0.1.67/batching.html
            autoround_num_ipus: When selecting multiple IPUs, auto-rounds to powers of 2 as required for IPUs.
            autoreport: Enable auto-reporting for IPUs using PopVision
                https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/graph/graph.html
            autoreport_dir: Optional directory to store autoReport output.
            convert_model_to_half: Converts the model to half precision, which can be used for pure FP16 training.
        """
        super().__init__(parallel_devices, cluster_environment)
        if not poptorch.ipuHardwareIsAvailable():
            raise MisconfigurationException(
                "The IPU Accelerator requires IPU devices to run. "
                "Learn more or get started with IPUs at https://www.graphcore.ai/getstarted"
            )

        self.convert_model_to_half = convert_model_to_half
        self.device_iterations = device_iterations
        self.autoround_num_ipus = autoround_num_ipus
        self.autoreport = autoreport
        self.autoreport_dir = autoreport_dir
        self.poptorch_models = {}
        self._original_accumulate_grad_batches = None

        if self.autoreport:
            options = {"autoReport.all": self.autoreport}
            if self.autoreport_dir:
                if not os.path.exists(self.autoreport_dir):
                    os.makedirs(self.autoreport_dir)
                options["autoReport.directory"] = self.autoreport_dir
            os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(options)

    @property
    def lightning_module(self) -> Optional[LightningModule]:
        return self.model.module if isinstance(self.model, LightningIPUModule) else self.model

    def pre_dispatch(self) -> None:
        self._handle_gradient_accumulation_steps()
        if self.convert_model_to_half:
            log.info('Using full 16bit precision, converting LightningModule weights to FP16.')
            self.model = self.model.half()
        precision = self.lightning_module.trainer.precision
        precision = 16 if self.convert_model_to_half else precision

        model = LightningIPUModule(self.lightning_module, precision)
        self.model = model

        # Separate models are instantiated for different stages, but they share the same weights on host.
        # When validation/test models are run, weights are synced first.

        if self.lightning_module.trainer.training:
            # Create model for training which will run training.
            optimizer = self.lightning_module.trainer.optimizers[0]
            model = poptorch.trainingModel(model=model, options=self._create_opts(training=True), optimizer=optimizer)
            self.poptorch_models['train'] = model
        for x in ('val', 'test', 'predict'):
            model = poptorch.inferenceModel(
                model=model,
                options=self._create_opts(training=False),
            )
            self.poptorch_models[x] = model

    @property
    def replication_factor(self):
        return len(self.parallel_devices)

    def _create_opts(self, training: bool):
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

    def on_reset_train_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        return self.process_dataloader(dataloader)

    def on_reset_val_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        return self.process_dataloader(dataloader)

    def on_reset_test_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        return self.process_dataloader(dataloader)

    def on_reset_predict_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
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

    def _handle_gradient_accumulation_steps(self):
        """
        This functions overrides the trainer.accumulation_scheduler to generate
        ``accumulate_grad_batches=1``.
        Therefore, ``optimizer_step`` will be called on every batch, and the IPU will handle grad accumulation.
        """
        self._original_accumulate_grad_batches = self.lightning_module.trainer.accumulate_grad_batches
        if not isinstance(self._original_accumulate_grad_batches, int):
            raise MisconfigurationException(
                f"IPUs currently only support accumulate_grad_batches being an integer value. "
                f"Received {self._original_accumulate_grad_batches}"
            )
        if self._original_accumulate_grad_batches > 1:
            self.lightning_module.trainer.accumulation_scheduler = GradientAccumulationScheduler({0: 1})

    def update_global_step(self, total_batch_idx: int, current_global_step: int) -> int:
        if self._original_accumulate_grad_batches > 1:
            if total_batch_idx % self._original_accumulate_grad_batches == 0:
                current_global_step += 1
            return current_global_step
        return super().update_global_step(total_batch_idx, current_global_step)

    @property
    def _n_replicate(self):
        # Ensure we replicate values to have enough dimensions to split across devices
        accumulate_grad_batches = self._original_accumulate_grad_batches
        return self.replication_factor * self.device_iterations * accumulate_grad_batches

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
        return self.poptorch_models['train'](*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        args = self._prepare_input(args)
        return self.poptorch_models['val'](*args, **kwargs)

    def test_step(self, *args, **kwargs):
        args = self._prepare_input(args)
        return self.poptorch_models['test'](*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        args = self._prepare_input(args)
        return self.poptorch_models['predict'](*args, **kwargs)

    def teardown(self) -> None:
        for k, model in self.poptorch_models.items():
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
        self._load_model('train')

    def on_validation_start(self):
        self._load_model('val')

    def on_test_start(self):
        self._load_model('test')

    def on_predict_start(self):
        self._load_model('predict')

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
        self.poptorch_models['train'].setOptimizer(optimizer)

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
