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
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Union

import torch.distributed as torch_distrib
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer

from pytorch_lightning import _logger as log
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.distributed import LightningDistributedModule, prepare_for_backward
from pytorch_lightning.plugins.legacy.plugin import LightningPlugin
from pytorch_lightning.utilities import DeviceType


class DDPPlugin(LightningPlugin):
    """
    Plugin to link a custom ddp implementation to any arbitrary accelerator.

    This plugin forwards all constructor arguments to :class:`~torch.nn.parallel.DistributedDataParallel`.

    Example::

        class MyDDP(DDPPlugin):

            def configure_ddp(self, model, device_ids):
                model = MyDDPWrapper(LightningDistributedModule(model), device_ids)
                return model

        my_ddp = MyDDP()
        trainer = Trainer(accelerator='ddp_x', plugins=[my_ddp])
    """

    def __init__(self, **kwargs):
        self._ddp_kwargs: Dict[str, Any] = kwargs

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ) -> DistributedDataParallel:
        """
        Pass through all customizations from constructor to :class:`~torch.nn.parallel.DistributedDataParallel`.
        Override to define a custom DDP implementation.

        .. note:: This requires that your DDP implementation subclasses
            :class:`~torch.nn.parallel.DistributedDataParallel` and that
            the original LightningModule gets wrapped by
            :class:`~pytorch_lightning.overrides.data_parallel.LightningDistributedModule`.

        The default implementation is::

            def configure_ddp(self, model, device_ids):
                model = DistributedDataParallel(
                    LightningDistributedModule(model),
                    device_ids=device_ids,
                    **self._ddp_kwargs,
                )
                return model

        Args:
            model: the LightningModule
            device_ids: the list of devices available

        Returns:
            the model wrapped in :class:`~torch.nn.parallel.DistributedDataParallel`

        """
        # if unset, default `find_unused_parameters` `True`
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get(
            "find_unused_parameters", True
        )
        model = DistributedDataParallel(
            module=LightningDistributedModule(model),
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
        # Todo: required argument `is_slurm_managing_tasks` is not used
        os.environ["MASTER_ADDR"] = str(cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(cluster_environment.world_size())
        torch_backend = "nccl" if trainer._device_type == DeviceType.GPU else "gloo"

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

        Returns:
            args moved to correct device if needed.
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
            model: Union[DistributedDataParallel, LightningModule]
    ) -> LightningModule:
        """
        Override to modify returning base :class:`LightningModule`
        when accessing variable and functions outside of the parallel wrapper.

        Example::
            ref_model = ddp_plugin.get_model_from_plugin(model)
            ref_model.training_step(...)

        Args:
            model: Model with parallel wrapper.

        Returns:
            Reference :class:`LightningModule` within parallel wrapper.

        """
        if isinstance(model, DistributedDataParallel):
            model = model.module
        if isinstance(model, LightningDistributedModule):
            model = model.module
        return model

    @contextmanager
    def block_backward_sync(self, model: DistributedDataParallel):
        """
        Blocks ddp sync gradients behaviour on backwards pass.
        This is useful for skipping sync when accumulating gradients, reducing communication overhead

        Returns:
            context manager with sync behaviour off
        """
        yield model.no_sync()

    def on_before_manual_backward(self, model: DistributedDataParallel, output: Any):
        prepare_for_backward(model, output)

    def distributed_sampler_kwargs(self, distributed_sampler_kwargs):
        return distributed_sampler_kwargs

    @property
    def data_parallel_group(self):
        """
        Return the group that this process exists in. By default, this is the world size.
        Useful for when additional parallel groups have been created, to select certain processes.

        Returns:
            The ProcessGroup this process exists in.
        """
        return torch_distrib.group.WORLD
