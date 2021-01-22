import os
from abc import ABC, abstractmethod
from typing import Optional

import torch

from pytorch_lightning import _logger as log
from pytorch_lightning.accelerators.plugins.base_plugin import Plugin


class TrainingTypePlugin(Plugin, ABC):
    def __init__(self):
        self._model = None
        self._results = None
        self.global_rank = 0

    @property
    @abstractmethod
    def on_gpu(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def root_device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def model_to_device(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def is_global_zero(self):
        raise NotImplementedError

    @abstractmethod
    def reduce(self, output, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def barrier(self, name: Optional[str] = None):
        raise NotImplementedError

    @abstractmethod
    def broadcast(self, obj: object, src: int = 0) -> object:
        raise NotImplementedError

    # TODO method this is currently unused
    def set_nvidia_flags(self, is_slurm_managing_tasks, device_ids):
        if device_ids is None:
            return

        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join([str(x) for x in range(torch.cuda.device_count())])
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        log.info(f'LOCAL_RANK: {self.trainer.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]')

    def reduce_early_stopping_decision(self, should_stop: bool) -> bool:
        return should_stop

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model

    @property
    def lightning_module(self):
        return self._model

    @property
    def results(self):
        """
        The results of the last training/testing run will be cached here.
        In distributed training, we make sure to transfer the results to the appropriate master process.
        """
        # TODO: improve these docs
        return self._results

    @property
    def rpc_enabled(self):
        return False

    def start_training(self, trainer):
        # double dispatch to initiate the training loop
        self._results = trainer.train()

    def start_testing(self, trainer):
        # double dispatch to initiate the test loop
        self._results = trainer.run_test()
