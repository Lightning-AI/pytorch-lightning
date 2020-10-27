from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.core.lightning import LightningModule
from typing import List


class DDPPlugin(object):
    """
    Plugin to link a custom ddp implementation to any arbitrary accelerator.

    Example::

        class MyDDP(DDPPlugin):

            def configure_ddp(self, model, device_ids):
                model = MyDDPWrapper(model, device_ids)
                return model

        my_ddp = MyDDP()
        trainer = Trainer(accelerator='ddp_x', plugins=[my_ddp])

    """

    def configure_ddp(self, model: LightningModule, device_ids: List[int]) -> LightningDistributedDataParallel:
        """
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
        model = LightningDistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=True)
        return model
