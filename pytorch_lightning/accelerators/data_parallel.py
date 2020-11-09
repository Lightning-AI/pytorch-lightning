from abc import ABC, abstractmethod

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


class ParallelPlugin(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def reduce(self, output):
        raise NotImplementedError

    @abstractmethod
    @property
    def root_device(self):
        raise NotImplementedError


class DataParallelPlugin(ParallelPlugin):
    def __init__(self, parallel_device_ids):
        super().__init__()
        self.parallel_device_ids = parallel_device_ids

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


class DistributedDataParallelPlugin(ParallelPlugin):
    def __init__(self, parallel_device_ids, num_nodes, num_processes, **ddp_kwargs):
        super().__init__(self)

        self.task_idx = None
        self._has_spawned_children = False
        self.interactive_ddp_procs = []
        self.dist = LightningDistributed()
        self.parallel_device_ids = parallel_device_ids
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
        if self.trainer.data_parallel_device_ids is None:
            raise MisconfigurationException("you selected (distribute_backend = ddp) but did not set Trainer(gpus=?)")

        os.environ["PL_TRAINER_GPUS"] = ",".join([str(i) for i in self.parallel_device_ids])
        os.environ["PL_IN_DDP_SUBPROCESS"] = "1"

        # TODO: Change t
        if self.trainer.logger is not None:
            os.environ["PL_EXP_VERSION"] = str(self.trainer.logger.version)

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
            self.trainer.global_rank,
            self.trainer.world_size,
            self.trainer.is_slurm_managing_tasks
        )

        # call setup after the ddp process has connected
        self.trainer.call_setup_hook(model)

        # on world_size=0 let everyone know training is starting
        if self.trainer.is_global_zero and not torch.distributed.is_initialized():
            log.info('-' * 100)
            log.info(f'distributed_backend={self.trainer.distributed_backend}')
            log.info(f'All DDP processes registered. Starting ddp with {self.trainer.world_size} processes')
            log.info('-' * 100)

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
        self.barrier('ddp_setup')
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
        torch_backend = "nccl" if self.trainer.on_gpu else "gloo"

        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch_distrib.init_process_group(torch_backend, rank=global_rank, world_size=world_size)

    def configure_ddp(
        self, model: LightningModule, device_ids: List[int]
    ) -> LightningDistributedDataParallel:
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
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get(
            "find_unused_parameters", True
        )
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

    def sync_tensor(self,
                    tensor: Union[torch.Tensor],
                    group: Optional[Any] = None,
                    reduce_op: Optional[Union[ReduceOp, str]] = None) -> torch.Tensor:
        """

        """
        return sync_ddp_if_available(tensor, group, reduce_op)
