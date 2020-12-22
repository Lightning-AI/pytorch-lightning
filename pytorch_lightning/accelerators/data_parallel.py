from abc import ABC, abstractmethod
import re
from contextlib import contextmanager

from pytorch_lightning.utilities.cloud_io import atomic_save, load as pl_load
from pytorch_lightning.accelerators.base_plugin import Plugin

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.distributed.dist import LightningDistributed
import torch
import os
from pytorch_lightning.core.step_result import Result
from typing import Any, Dict, List, Optional, Union
from pytorch_lightning.overrides.data_parallel import LightningDataParallel, LightningDistributedDataParallel
import sys
from os.path import abspath
from time import sleep
import subprocess
from pytorch_lightning.utilities.distributed import find_free_network_port, rank_zero_only
import numpy as np
import torch.distributed as torch_distrib
from pytorch_lightning import _logger as log
import torch.multiprocessing as mp
from pytorch_lightning.utilities.distributed import sync_ddp_if_available, rank_zero_warn, rank_zero_info

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

    def determine_local_rank(self):
        return int(os.environ.get('LOCAL_RANK', 0))

    def determine_node_rank(self):
        # torchelastic uses the envvar GROUP_RANK, whereas other systems(?) use NODE_RANK.
        # otherwise use given node rank or default to node rank 0
        env_vars = ['NODE_RANK', 'GROUP_RANK']
        node_ids = [(k, os.environ.get(k, None)) for k in env_vars]
        node_ids = [(k, v) for k, v in node_ids if v is not None]
        if len(node_ids) == 0:
            return 0
        if len(node_ids) > 1:
            log.warning(f"Multiple environment variables ({node_ids}) defined for node rank. Using the first one.")
        k, rank = node_ids.pop()
        rank_zero_info(f"Using environment variable {k} for node rank ({rank}).")
        return int(rank)

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


class SingleDevicePlugin(TrainingTypePlugin):
    def __init__(self, device):
        super().__init__()
        self.device: torch.device = device

    @property
    def on_gpu(self):
        return self.device.type == "cuda" and torch.cuda.is_available()

    def reduce(self, output, *args, **kwargs):
        return output

    @property
    def root_device(self):
        return self.device
    
    def model_to_device(self):
        if self.on_gpu:
            torch.cuda.set_device(self.root_device)

        self._model.to(self.root_device)

    def connect(self, model: torch.nn.Module):
        self._model = model
        self.model_to_device()
        return self.model

    @property
    def is_global_zero(self):
        return True

    def barrier(self, *args, **kwargs):
        pass

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj


class ParallelPlugin(TrainingTypePlugin, ABC):
    def __init__(self, parallel_devices: List[torch.device], cluster_environment=None):
        super().__init__()
        self.parallel_devices = parallel_devices
        self.local_rank = 0
        self.world_size = 1
        self.cluster_environment = cluster_environment

    @property
    @abstractmethod
    def root_device(self):
        raise NotImplementedError

    @property
    def on_gpu(self):
        return self.root_device.type == "cuda" and torch.cuda.is_available()

    @abstractmethod
    def setup(self, model):
        raise NotImplementedError

    def connect(self, model, *args, **kwargs):
        self.setup(model)
        return self.model

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(
            num_replicas=len(self.parallel_devices),
            rank=self.global_rank
        )
        return distributed_sampler_kwargs

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
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    @contextmanager
    def block_backward_sync(self):
        """
        Blocks ddp sync gradients behaviour on backwards pass.
        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        if isinstance(self.model, LightningDistributedDataParallel):
            yield self.model.no_sync()
        else:
            yield None


class DataParallelPlugin(ParallelPlugin):

    def setup(self, model):
        self._model = LightningDataParallel(model, self.parallel_devices)

    def reduce(self, output, *args, **kwargs):
        if isinstance(output, Result):
            output.dp_reduce()

        elif isinstance(output, torch.Tensor):
            output = output.mean()

        return output

    @property
    def root_device(self):
        return self.parallel_devices[0]

    @property
    def lightning_module(self):
        return self._model.module

    def model_to_device(self):
        # no need to do anything when model is wrapped in torch.nn.DataParallel
        pass

    def barrier(self, *args, **kwargs):
        pass

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj


class DDPPlugin(ParallelPlugin):

    distributed_backend = "ddp"

    def __init__(
            self,
            parallel_devices,
            num_nodes=1,
            cluster_environment=None,
            is_slurm_managing_tasks=False,
            **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(parallel_devices=parallel_devices, cluster_environment=cluster_environment)
        self.interactive_ddp_procs = []
        self.num_nodes = num_nodes
        self.is_slurm_managing_tasks = is_slurm_managing_tasks
        self.dist = LightningDistributed()
        self._ddp_kwargs = kwargs
        self._has_spawned_children = False
        self.task_idx = None
        self.num_processes = len(parallel_devices)

    @property
    def root_device(self):
        return self.parallel_devices[self.local_rank]

    @property
    def lightning_module(self):
        # the model may not be wrapped with DistributedDataParallel if calling this too early
        return getattr(self._model, "module", self._model)

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(
            num_replicas=(self.num_nodes * self.num_processes),
            rank=self.global_rank
        )
        return distributed_sampler_kwargs

    def determine_local_rank(self):
        if self.is_slurm_managing_tasks:
            return int(os.environ['SLURM_LOCALID'])
        else:
            return super().determine_node_rank()

    def determine_node_rank(self):
        if self.is_slurm_managing_tasks:
            return int(os.environ['SLURM_NODEID'])
        else:
            return super().determine_node_rank()

    def setup(self, model):
        self._model = model

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
        if self.parallel_devices is None:
            raise MisconfigurationException("you selected (distribute_backend = ddp) but did not set Trainer(gpus=?)")

        os.environ["PL_TRAINER_GPUS"] = ",".join([str(device.index) for device in self.parallel_devices])
        os.environ["PL_IN_DDP_SUBPROCESS"] = "1"

        if self.lightning_module.logger is not None:
            os.environ["PL_EXP_VERSION"] = str(self.lightning_module.logger.version)

        num_gpus = len(self.parallel_devices)
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

    def _check_can_spawn_children(self):
        if self._has_spawned_children:
            raise RuntimeError(
                "You tried to run `.fit` or `.test` multiple times in the same script."
                " This is not supported in DDP mode, switch to `distributed_backend='ddp_spawn'` instead."
            )

    def set_world_ranks(self):
        self.local_rank = self.task_idx
        # TODO: check from where we get node_rank and num_processes
        self.global_rank = self.determine_node_rank() * self.num_processes + self.task_idx
        self.world_size = self.num_nodes * self.num_processes

    def configure_ddp(self):
        # if unset, default `find_unused_parameters` `True`
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)
        self._model = LightningDistributedDataParallel(
            self.model,
            device_ids=self.determine_ddp_device_ids(),
            **self._ddp_kwargs,
        )

    def determine_ddp_device_ids(self):
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

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
        # TODO: check if needed
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        # determine which process we are and world size
        self.set_world_ranks()

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        # TODO: CHeck is_slurm_managing_tasks
        self.init_ddp_connection(self.global_rank, self.world_size)

        # TODO: Move this somewhere else
        # self.trainer.call_setup_hook(self.model)

        # on world_size=0 let everyone know training is starting
        if self.is_global_zero and not torch.distributed.is_initialized():
            log.info("-" * 100)
            log.info(f"distributed_backend={self.distributed_backend}")
            log.info(f"All DDP processes registered. Starting ddp with {self.world_size} processes")
            log.info("-" * 100)

        # set the ranks and devices
        self.dist.rank = self.global_rank
        self.dist.device = self.root_device

        self.model = self.configure_sync_batchnorm(self.model)

        # move the model to the correct device
        self.model_to_device()

        self.configure_ddp()

        self.barrier()

    def post_training(self):
        if "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]

    def barrier(self, *args, **kwargs):
        if torch_distrib.is_initialized():
            torch_distrib.barrier()

    def broadcast(self, obj: object, src: int = 0) -> object:
        return self.dist.broadcast(obj)

    def model_to_device(self):
        if self.root_device.type == "cuda":
            torch.cuda.set_device(self.root_device)
        self.model.to(self.root_device)

    def reduce(self, output, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None):
        if isinstance(output, torch.Tensor):
            output = sync_ddp_if_available(output, group, reduce_op)
        return output


class DDPSpawnPlugin(ParallelPlugin):

    distributed_backend = "ddp_spawn"

    def __init__(
        self,
        parallel_devices,
        num_nodes=1,
        cluster_environment=None,
        is_slurm_managing_tasks=False,
        **kwargs: Dict[str, Any]
    ):
        super().__init__(parallel_devices=parallel_devices, cluster_environment=cluster_environment)
        self.num_nodes = num_nodes
        self.is_slurm_managing_tasks = is_slurm_managing_tasks
        self._ddp_kwargs = kwargs
        self.dist = LightningDistributed()
        self.num_processes = len(parallel_devices)
        self.mp_queue = None

    @property
    def root_device(self):
        return self.parallel_devices[self.local_rank]

    @property
    def lightning_module(self):
        # the model may not be wrapped with DistributedDataParallel if calling this too early
        return getattr(self._model, "module", self._model)

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(
            num_replicas=(self.num_nodes * self.num_processes),
            rank=self.global_rank
        )
        return distributed_sampler_kwargs

    def setup(self, model):
        self._model = model

        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(find_free_network_port()))

        # pass in a state q
        smp = mp.get_context('spawn')
        self.mp_queue = smp.SimpleQueue()

    def set_world_ranks(self, process_idx):
        self.local_rank = process_idx
        # check from where we get node_rank, num_processes and num_nodes
        self.global_rank = self.determine_node_rank() * self.num_processes + process_idx
        self.world_size = self.num_nodes * self.num_processes

    def start_training(self, trainer):
        mp.spawn(self.new_process, nprocs=self.num_processes, args=(trainer,))

    def start_testing(self, trainer):
        mp.spawn(self.new_process, nprocs=self.num_processes, args=(trainer, ))

    def new_process(self, process_idx, trainer):
        # TODO: check if needed
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        self.set_world_ranks(process_idx)

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        # TODO: CHeck is_slurm_managing_tasks
        self.init_ddp_connection(self.global_rank, self.world_size)

        # TODO: Move this somewhere else
        # self.trainer.call_setup_hook(self.model)

        # on world_size=0 let everyone know training is starting
        if self.is_global_zero and not torch.distributed.is_initialized():
            log.info("-" * 100)
            log.info(f"distributed_backend={self.distributed_backend}")
            log.info(f"All DDP processes registered. Starting ddp with {self.world_size} processes")
            log.info("-" * 100)

        # set the ranks and devices
        self.dist.rank = self.global_rank
        self.dist.device = self.root_device

        self.model = self.configure_sync_batchnorm(self.model)

        # move the model to the correct device
        self.model_to_device()

        self.configure_ddp()

        self.barrier()

        if trainer.testing:
            results = trainer.run_test()
        else:
            results = trainer.train()

        # persist info in ddp_spawn
        self.transfer_distrib_spawn_state_on_fit_end(results)

    def post_training(self):
        # restore main state with best weights
        best_path = self.mp_queue.get()
        last_path = self.mp_queue.get()
        self._results = self.mp_queue.get()

        # recover the weights of the processes trained in the children
        self.__recover_child_process_weights(best_path, last_path)

    def configure_ddp(self):
        # if unset, default `find_unused_parameters` `True`
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)
        self.model = LightningDistributedDataParallel(
            self.model,
            device_ids=self.determine_ddp_device_ids(),
            **self._ddp_kwargs,
        )

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        # TODO: this code is duplicated in DDP and DDPSpawn, make this a function
        os.environ["MASTER_ADDR"] = str(self.cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(self.cluster_environment.world_size())
        torch_backend = "nccl" if self.on_gpu else "gloo"

        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch_distrib.init_process_group(torch_backend, rank=global_rank, world_size=world_size)

    def determine_ddp_device_ids(self):
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

    def transfer_distrib_spawn_state_on_fit_end(self, results):
        # TODO: is there a better way than accessing callback through model -> trainer -> callback?
        best_model_path = self.lightning_module.trainer.checkpoint_callback.best_model_path

        if self.global_rank == 0 and self.mp_queue is not None:
            rank_zero_warn('cleaning up ddp environment...')

            # save the last weights
            last_path = None
            # TODO: is there a better way than accessing trainer through model -> trainer?
            if not self.lightning_module.trainer.testing and best_model_path is not None and len(best_model_path) > 0:
                last_path = re.sub('.ckpt', '.tmp_end.ckpt', best_model_path)
                atomic_save(self.lightning_module.state_dict(), last_path)

            # todo, pass complete checkpoint as state dictionary
            self.mp_queue.put(best_model_path)
            self.mp_queue.put(last_path)
            self.mp_queue.put(results)

    def __recover_child_process_weights(self, best_path, last_path):
        # TODO: is there a better way than accessing callback through model -> trainer -> callback?
        # transfer back the best path to the trainer
        if self.lightning_module.trainer.checkpoint_callback:
            self.lightning_module.trainer.checkpoint_callback.best_model_path = best_path
        # todo, pass also best score

        # load last weights
        # TODO: How to get self.trainer.testing?
        if last_path is not None: # and not self.trainer.testing:
            ckpt = pl_load(last_path, map_location=lambda storage, loc: storage)
            self.lightning_module.load_state_dict(ckpt)

    def determine_local_rank(self):
        if self.is_slurm_managing_tasks:
            return int(os.environ['SLURM_LOCALID'])
        else:
            return super().determine_node_rank()

    def determine_node_rank(self):
        if self.is_slurm_managing_tasks:
            return int(os.environ['SLURM_NODEID'])
        else:
            return super().determine_node_rank()

    def barrier(self, *args, **kwargs):
        if torch_distrib.is_initialized():
            torch_distrib.barrier()

    def broadcast(self, obj: object, src: int = 0) -> object:
        return self.dist.broadcast(obj)

    def model_to_device(self):
        if self.root_device.type == "cuda":
            torch.cuda.set_device(self.root_device)
        self.model.to(self.root_device)

    def reduce(self, output, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None):
        if isinstance(output, torch.Tensor):
            output = sync_ddp_if_available(output, group, reduce_op)
        return output


# TODO: DDP2 (?), HOROVOD DDP AND HPC DDP
