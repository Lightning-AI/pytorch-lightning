import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Union

import torch
import torch.distributed as torch_distrib
from torch.optim import Optimizer

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.plugins.plugin import LightningPlugin

log = logging.getLogger(__name__)


class DDPPlugin(LightningPlugin):
    """
    Plugin to link a custom ddp implementation to any arbitrary accelerator.

    This plugin forwards all constructor arguments to `LightningDistributedDataParallel`,
    which in turn forwards all args to `DistributedDataParallel`.

    Example::

        class MyDDP(DDPPlugin):

            def configure_ddp(self, model, device_ids):
                model = MyDDPWrapper(model, device_ids)
                return model

        my_ddp = MyDDP()
        trainer = Trainer(accelerator='ddp_x', plugins=[my_ddp])
    """

    def __init__(self, **kwargs):
        self._ddp_kwargs: Dict[str, Any] = kwargs

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
                    model, device_ids=device_ids, find_unused_parameters=False
                )
                return model

        Args:
            model: the lightningModule
            device_ids: the list of devices available

        Returns:
            the model wrapped in LightningDistributedDataParallel

        """
        # if unset, default `find_unused_parameters` `False`
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get(
            "find_unused_parameters", False
        )
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            **self._ddp_kwargs,
        )
        return model

    def init_ddp_connection(
            self,
            trainer,
            cluster_environment,
            global_rank: int,
            world_size: int,
            is_slurm_managing_tasks: bool = True,
    ) -> None:
        os.environ["MASTER_ADDR"] = str(cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(cluster_environment.world_size())
        torch_backend = "nccl" if trainer.on_gpu else "gloo"

        if not torch_distrib.is_initialized():
            log.info(
                f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}"
            )
            torch_distrib.init_process_group(
                torch_backend, rank=global_rank, world_size=world_size
            )

    @property
    def is_running_single_process_per_device(self) -> bool:
        # objects do not need to be scattered in single process per device, move objects upfront to device
        # This property is used in ``self.on_before_forward`` function.
        return self.device_ids is not None and len(self.device_ids) == 1

    def on_before_forward(self, model: LightningModule, *args):
        """
        Override to handle custom edge case.

        Args:
            args: Inputs to the model.
            model: Model to train.
        Returns: args moved to correct device if needed.
        """
        if self.is_running_single_process_per_device:
            args = model.transfer_batch_to_device(args, model.device)
        return args

    def optimizer_state(self, optimizer: Optimizer) -> dict:
        return optimizer.state_dict()

    def on_after_setup_optimizers(self, trainer):
        """
        Called after optimizers have been set-up. This is useful for doing any configuration options in RPC, or
        state sharding.
        """

    def get_model_from_plugin(
            self,
            model: Union[LightningDistributedDataParallel, LightningModule]
    ) -> LightningModule:
        """
        Override to modify returning base :class:`LightningModule`
        when accessing variable and functions outside of the parallel wrapper.

        Example::
            ref_model = ddp_plugin.get_model_from_plugin(model)
            ref_model.training_step(...)

        Args:
            model: Model with parallel wrapper.

        Returns: Reference :class:`LightningModule` within parallel wrapper.

        """
        if isinstance(model, LightningDistributedDataParallel):
            return model.module
        return model

    @contextmanager
    def block_backward_sync(self, model: LightningDistributedDataParallel):
        """
        Blocks ddp sync gradients behaviour on backwards pass.
        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        yield model.no_sync()

    def on_before_manual_backward(self, model: LightningDistributedDataParallel, output: Any):
        model.reducer_prepare_for_backwards(output)

    def on_after_manual_backward(self, model: LightningDistributedDataParallel):
        model.reducer_reset_hooks()

    def distributed_sampler_kwargs(self, distributed_sampler_kwargs):
        return distributed_sampler_kwargs

    @property
    def data_parallel_group(self):
        """
        Return the group that this process exists in. By default, this is the world size.
        Useful for when additional parallel groups have been created, to select certain processes.
        Returns: The ProcessGroup this process exists in.
        """
        return torch_distrib.group.WORLD
