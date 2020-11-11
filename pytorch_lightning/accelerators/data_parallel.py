from abc import ABC, abstractmethod
from contextlib import contextmanager
from os import stat
from pytorch_lightning.accelerators.base_plugin import Plugin

from torch.nn.parallel.distributed import DistributedDataParallel
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.distributed.dist import LightningDistributed
import torch
import os
from pytorch_lightning.core.step_result import Result
from typing import Any, Dict, List, Optional, Union
from pytorch_lightning.overrides.data_parallel import LightningDataParallel, LightningDistributedDataParallel
from torch.nn.parallel.data_parallel import DataParallel
import sys
from os.path import abspath
from time import sleep
import subprocess
from pytorch_lightning.utilities.distributed import find_free_network_port, rank_zero_only
import numpy as np
import torch.distributed as torch_distrib
from pytorch_lightning import _logger as log
import contextlib
import torch.multiprocessing as mp

from pytorch_lightning.utilities.distributed import sync_ddp_if_available

try:
    from hydra.utils import to_absolute_path, get_original_cwd
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HYDRA_AVAILABLE = False
else:
    HYDRA_AVAILABLE = True

if torch.distributed.is_available():
    from torch.distributed import ReduceOp
else:

    class ReduceOp:
        SUM = None


class TrainingTypePlugin(Plugin, ABC):
    def __init__(self, logger=None):
        self.model = None
        self.global_rank = 0
        self.logger = logger

    @abstractmethod
    @property
    def on_gpu(self):
        raise NotImplementedError

    @abstractmethod
    @property
    def root_device(self):
        raise NotImplementedError

    @abstractmethod
    def model_to_device(self):
        raise NotImplementedError

    @abstractmethod
    @property
    def is_global_zero(self):
        raise NotImplementedError

    @abstractmethod
    def barrier(self):
        raise NotImplementedError

class SingleDevicePlugin(TrainingTypePlugin):
    def __init__(self, device, logger=None):
        super().__init__(logger=logger)
        self.device: torch.device = device

    @property
    def on_gpu(self):
        return self.device.type == "cuda" and torch.cuda.is_available()

    def reduce(self, output):
        return output

    @property
    def root_device(self):
        return self.device
    
    def model_to_device(self):
        self.model.to(self.root_device)

    def connect(self, model: torch.nn.Module, optimizers, lr_schedulers):
        self.model = model

    @property
    def is_global_zero(self):
        return True

    def barrier(self):
        pass

    

class ParallelPlugin(TrainingTypePlugin, ABC):
    def __init__(self, parallel_device_ids, logger=None, cluster_environment=None):
        super().__init__(logger=logger)
        self.parallel_device_ids = parallel_device_ids
        self.local_rank = 0
        self.world_size = 1
        self.cluster_environment = cluster_environment

    @abstractmethod
    def reduce(self, output):
        raise NotImplementedError

    @abstractmethod
    @property
    def root_device(self):
        raise NotImplementedError

    @property
    def on_gpu(self):
        return self.parallel_device_ids and torch.cuda.is_available()

    @abstractmethod
    def setup(self, model):
        raise NotImplementedError

    def connect(self, model):
        self.setup(model)

        return self.model

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0


class DataParallelPlugin(ParallelPlugin):
    def setup(self, model):
        self.model = LightningDataParallel(model, self.parallel_device_ids)

    def reduce(self, output):
        if isinstance(output, Result):
            output.dp_reduce()

        elif isinstance(output, torch.Tensor):
            output = output.mean()

        return output

    @property
    def root_device(self):
        return self.parallel_device_ids[0]

    def barrier(self):
        pass


class DDPPlugin(ParallelPlugin):

    distributed_backend = "ddp"

    def __init__(self, parallel_device_ids, logger=None, cluster_environment=None) -> None:
        super().__init__(parallel_device_ids=parallel_device_ids, logger=logger, cluster_environment=cluster_environment)
        self._has_spawned_children = False
        self.interactive_ddp_procs = []
        self.dist = LightningDistributed()

    @property
    def root_device(self):
        return self.parallel_device_ids[self.local_rank]

    def setup(self, model):

        self.model = model

        # start the other scripts
        if os.environ.get("PL_IN_DDP_SUBPROCESS", "0") != "1":
            self._call_children_scripts()

        # set the task idx
        self.task_idx = int(os.environ["LOCAL_RANK"])

    def _call_children_scripts(self):

        # bookkeeping of spawned processes
        assert self.global_rank == 0
        self._check_can_spawn_children()
        self._has_spawned_children = True

        # DDP Environment variables
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(find_free_network_port()))

        # allow the user to pass the node rank
        node_rank = "0"
        node_rank = os.environ.get("NODE_RANK", node_rank)
        node_rank = os.environ.get("GROUP_RANK", node_rank)
        os.environ["NODE_RANK"] = node_rank
        os.environ["LOCAL_RANK"] = "0"

        # when user is using hydra find the absolute path
        path_lib = abspath if not HYDRA_AVAILABLE else to_absolute_path

        # pull out the commands used to run the script and resolve the abs file path
        command = sys.argv
        try:
            full_path = path_lib(command[0])
        except Exception as e:
            full_path = abspath(command[0])

        command[0] = full_path
        # use the same python interpreter and actually running
        command = [sys.executable] + command

        # the visible devices tell us how many GPUs we want to use.
        # when the trainer script was called the device has already been scoped by the time
        # code reaches this point. so, to call the scripts, we need to leave cuda visible devices alone
        # but forward the GPUs selected via environment variables
        if self.parallel_device_ids is None:
            raise MisconfigurationException("you selected (distribute_backend = ddp) but did not set Trainer(gpus=?)")

        os.environ["PL_TRAINER_GPUS"] = ",".join([str(i) for i in self.parallel_device_ids])
        os.environ["PL_IN_DDP_SUBPROCESS"] = "1"

        if self.logger is not None:
            os.environ["PL_EXP_VERSION"] = str(self.logger.version)

        num_gpus = len(self.data_parallel_device_ids)
        # TODO: Add num_nodes (pass it in?)
        os.environ["WORLD_SIZE"] = f"{num_gpus * self.num_nodes}"

        self.interactive_ddp_procs = []

        # TODO: Add num_processes (pass it in?)
        for local_rank in range(1, self.num_processes):
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{local_rank}"

            # remove env var if global seed not set
            if os.environ.get("PL_GLOBAL_SEED") is None and "PL_GLOBAL_SEED" in env_copy:
                del env_copy["PL_GLOBAL_SEED"]

            # start process
            # if hydra is available and initialized, make sure to set the cwd correctly
            cwd: Optional[str] = None
            if HYDRA_AVAILABLE:
                if HydraConfig.initialized():
                    cwd = get_original_cwd()
            proc = subprocess.Popen(command, env=env_copy, cwd=cwd)
            self.interactive_ddp_procs.append(proc)

            # starting all processes at once can cause issues
            # with dataloaders delay between 1-10 seconds
            delay = np.random.uniform(1, 5, 1)[0]
            sleep(delay)

    def _check_can_spawn_children(self):
        if self._has_spawned_children:
            raise RuntimeError(
                "You tried to run `.fit` or `.test` multiple times in the same script."
                " This is not supported in DDP mode, switch to `distributed_backend='ddp_spawn'` instead."
            )

    def set_world_ranks(self):
        self.local_rank = self.task_idx
        self.global_rank = self.node_rank * self.num_processes + self.task_idx
        self.world_size = self.num_nodes * self.num_processes

    def configure_ddp(self):
        # if unset, default `find_unused_parameters` `True`
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)
        self.model = LightningDistributedDataParallel(
            self.model,
            device_ids=self.determine_ddp_device_ids(),
            **self._ddp_kwargs,
        )

    def determine_ddp_device_ids(self):
        return [self.root_device]

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        # TODO: From where to get cluster environment?
        os.environ["MASTER_ADDR"] = str(self.cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(self.cluster_environment.world_size())
        torch_backend = "nccl" if self.on_gpu else "gloo"

        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch_distrib.init_process_group(torch_backend, rank=global_rank, world_size=world_size)

    def pre_training(self):
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        # show progressbar only on progress_rank 0
        # TODO: check where to move this. Cannot stay here, since we won't have access to progressbar here
        if (self.node_rank != 0 or self.task_idx != 0) and self.trainer.progress_bar_callback is not None:
            self.trainer.progress_bar_callback.disable()

        # determine which process we are and world size
        self.set_world_ranks()

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # TODO: This has to be done somewhere else!
        self.model.trainer = self.trainer

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.init_ddp_connection(self.global_rank, self.world_size, self.is_slurm_managing_tasks)

        # on world_size=0 let everyone know training is starting
        if self.is_global_zero and not torch.distributed.is_initialized():
            log.info("-" * 100)
            log.info(f"distributed_backend={self.distributed_backend}")
            log.info(f"All DDP processes registered. Starting ddp with {self.world_size} processes")
            log.info("-" * 100)

        self.model = self.configure_sync_batchnorm(self.model)

        # move the model to the correct device
        self.model_to_device()

        self.configure_ddp()

        self.barrier()

    def post_training(self):
        torch.cuda.empty_cache()

        if "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]

    @staticmethod
    def configure_sync_batchnorm(model: LightningModule) -> LightningModule:
        """
        Add global batchnorm for a model spread across multiple GPUs and nodes.

        Override to synchronize batchnorm between specific process groups instead
        of the whole world or use a different sync_bn like `apex`'s version.

        Args:
            model: pointer to current :class:`LightningModule`.

        Return:
            LightningModule with batchnorm layers synchronized between process groups
        """
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=None)

        return model

    def barrier(self):
        if torch_distrib.is_initialized():
            torch_distrib.barrier()

    def model_to_device(self):
        # TODO: Can we easily make this a property that falls back here?
        # self.trainer.root_gpu = self.trainer.data_parallel_device_ids[self.trainer.local_rank]
        torch.cuda.set_device(self.root_device)
        self.model.cuda(self.root_device)

    def reduce(self, output, group: Optional[Any] = None,
                    reduce_op: Optional[Union[ReduceOp, str]] = None):

        if isinstance(output, torch.Tensor):
            output = sync_ddp_if_available(output, group, reduce_op)
        
        return output

class DDPSpawnPlugin(ParallelPlugin):
    def __init__(self, parallel_device_ids, logger=None, cluster_environment=None):
        super().__init__(parallel_device_ids=parallel_device_ids, logger=logger, cluster_environment=cluster_environment)

        self.dist = LightningDistributed()
        # TODO: how to get in nprocs? probably pass it
        self.nprocs = nprocs
        self.mp_queue = None

    def setup(self, model):
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(find_free_network_port()))

        # pass in a state q
        smp = mp.get_context('spawn')
        self.mp_queue = smp.SimpleQueue()

    def pre_training(self, process_idx = None, mp_queue=None, ):
        # TODO: use a mixture of os.fork and multiprocesing queue for ddp here
        os.fork()

    
















class MidDistributedDataParallelPlugin(ParallelPlugin):
    def __init__(self, parallel_device_ids, num_nodes, num_processes, **ddp_kwargs):
        super().__init__(parallel_device_ids)

        self.task_idx = None
        self._has_spawned_children = False
        self.interactive_ddp_procs = []
        self.dist = LightningDistributed()
        self.num_nodes = num_nodes
        self.num_processes = num_processes
        self._ddp_kwargs: Dict[str, Any] = ddp_kwargs

    def setup(self, model):
        # start the other scripts
        if os.environ.get("PL_IN_DDP_SUBPROCESS", "0") != "1":
            self._call_children_scripts()

        # set the task idx
        self.task_idx = int(os.environ["LOCAL_RANK"])

    def _call_children_scripts(self):
        assert self.trainer.global_rank == 0
        self._check_can_spawn_children()
        self._has_spawned_children = True

        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(find_free_network_port()))

        # allow the user to pass the node rank
        node_rank = "0"
        node_rank = os.environ.get("NODE_RANK", node_rank)
        node_rank = os.environ.get("GROUP_RANK", node_rank)
        os.environ["NODE_RANK"] = node_rank
        os.environ["LOCAL_RANK"] = "0"

        # when user is using hydra find the absolute path
        path_lib = abspath if not HYDRA_AVAILABLE else to_absolute_path

        # pull out the commands used to run the script and resolve the abs file path
        command = sys.argv
        try:
            full_path = path_lib(command[0])
        except Exception as e:
            full_path = abspath(command[0])

        command[0] = full_path
        # use the same python interpreter and actually running
        command = [sys.executable] + command

        # the visible devices tell us how many GPUs we want to use.
        # when the trainer script was called the device has already been scoped by the time
        # code reaches this point. so, to call the scripts, we need to leave cuda visible devices alone
        # but forward the GPUs selected via environment variables
        if self.parallel_device_ids is None:
            raise MisconfigurationException("you selected (distribute_backend = ddp) but did not set Trainer(gpus=?)")

        os.environ["PL_TRAINER_GPUS"] = ",".join([str(i) for i in self.parallel_device_ids])
        os.environ["PL_IN_DDP_SUBPROCESS"] = "1"

        if self.logger is not None:
            os.environ["PL_EXP_VERSION"] = str(self.logger.version)

        num_gpus = len(self.parallel_device_ids)
        os.environ["WORLD_SIZE"] = f"{num_gpus * self.num_nodes}"

        self.interactive_ddp_procs = []
        for local_rank in range(1, self.num_processes):
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{local_rank}"

            # remove env var if global seed not set
            if os.environ.get("PL_GLOBAL_SEED") is None and "PL_GLOBAL_SEED" in env_copy:
                del env_copy["PL_GLOBAL_SEED"]

            # start process
            # if hydra is available and initialized, make sure to set the cwd correctly
            cwd: Optional[str] = None
            if HYDRA_AVAILABLE:
                if HydraConfig.initialized():
                    cwd = get_original_cwd()
            proc = subprocess.Popen(command, env=env_copy, cwd=cwd)
            self.interactive_ddp_procs.append(proc)

            # starting all processes at once can cause issues
            # with dataloaders delay between 1-10 seconds
            delay = np.random.uniform(1, 5, 1)[0]
            sleep(delay)

    def barrier(self, name: Optional[str] = None):
        if torch_distrib.is_initialized():
            torch_distrib.barrier()

    # TODO: Refactor This! Not sure we still need the whole method here. Should be dione with some additional setup and cleaning logic
    def ddp_train(self, process_idx, model):
        """
        Entry point for ddp

        Args:
            process_idx:
            mp_queue: multiprocessing queue
            model:

        Returns:
            Dict with evaluation results

        """
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        # TODO: move this somewhere else!
        # show progressbar only on progress_rank 0
        if (self.trainer.node_rank != 0 or process_idx != 0) and self.trainer.progress_bar_callback is not None:
            self.trainer.progress_bar_callback.disable()

        # determine which process we are and world size
        self.set_world_ranks(process_idx)

        # set warning rank
        rank_zero_only.rank = self.trainer.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        model.trainer = self.trainer
        self.init_ddp_connection(
            self.trainer.global_rank, self.trainer.world_size, self.trainer.is_slurm_managing_tasks
        )

        # call setup after the ddp process has connected
        self.trainer.call_setup_hook(model)

        # on world_size=0 let everyone know training is starting
        if self.trainer.is_global_zero and not torch.distributed.is_initialized():
            log.info("-" * 100)
            log.info(f"distributed_backend={self.trainer.distributed_backend}")
            log.info(f"All DDP processes registered. Starting ddp with {self.trainer.world_size} processes")
            log.info("-" * 100)

        # call sync_bn before .cuda(), configure_apex and configure_ddp
        if self.trainer.sync_batchnorm:
            model = self.configure_sync_batchnorm(model)

        # move the model to the correct device
        self.model_to_device(model, process_idx)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.setup_optimizers(model)

        # set model properties before going into wrapper
        self.trainer.model_connector.copy_trainer_model_properties(model)

        # 16-bit
        model = self.trainer.precision_connector.connect(model)

        # device ids change depending on the DDP setup
        device_ids = self.get_device_ids()

        # allow user to configure ddp
        model = self.configure_ddp(model, device_ids)

        # set up training routine
        self.barrier("ddp_setup")
        self.trainer.train_loop.setup_training(model)

        # train or test
        results = self.train_or_test()

        # clean up memory
        torch.cuda.empty_cache()

        return results

    def init_ddp_connection(self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True) -> None:
        os.environ["MASTER_ADDR"] = str(self.cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(self.cluster_environment.world_size())
        torch_backend = "nccl" if self.on_gpu else "gloo"

        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch_distrib.init_process_group(torch_backend, rank=global_rank, world_size=world_size)

    def configure_ddp(self, model: LightningModule, device_ids: List[int]) -> LightningDistributedDataParallel:
        """
        Pass through all customizations from constructor to `LightningDistributedDataParallel`.
        Override to define a custom DDP implementation.

        .. note:: Only requirement is that your DDP implementation subclasses LightningDistributedDataParallel


        The default implementation is::

            def configure_ddp(self, model, device_ids):
                model = LightningDistributedDataParallel(
                    model, device_ids=device_ids, find_unused_parameters=True
                )
                return model

        Args:
            model: the lightningModule
            device_ids: the list of devices available

        Returns:
            the model wrapped in LightningDistributedDataParallel

        """
        # if unset, default `find_unused_parameters` `True`
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            **self._ddp_kwargs,
        )
        return model

    def configure_sync_batchnorm(self, model: LightningModule) -> LightningModule:
        """
        Add global batchnorm for a model spread across multiple GPUs and nodes.

        Override to synchronize batchnorm between specific process groups instead
        of the whole world or use a different sync_bn like `apex`'s version.

        Args:
            model: pointer to current :class:`LightningModule`.

        Return:
            LightningModule with batchnorm layers synchronized between process groups
        """
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=None)

        return model

    def sync_tensor(
        self, tensor: Union[torch.Tensor], group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
    ) -> torch.Tensor:
        """"""
        return sync_ddp_if_available(tensor, group, reduce_op)
