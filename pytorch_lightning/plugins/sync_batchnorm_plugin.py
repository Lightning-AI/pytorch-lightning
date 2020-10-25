import torch
from pytorch_lightning.core.lightning import LightningModule


class SyncBatchNormPlugin(object):
    """
    Plugin to link a custom sync batchnorm implementation to any arbitrary accelerator.
    Example::
        class MySyncBN(SyncBatchNormPlugin):

            def configure_sync_batchnorm(self, model):
                model = MySyncBNWrapper(model)
                return model
        my_sync_bn = MySyncBN()
        trainer = Trainer(sync_batchnorm=True, plugins=[my_sync_bn])
    """

    def configure_sync_batchnorm(self, model: LightningModule) -> LightningModule:
        """
        By default: adds global batchnorm for a model spread across multiple GPUs and nodes.
        Override this to synchronize batchnorm between specific process groups instead
        of the whole world or use a different sync_bn like `apex`'s version.
        Args:
            model: pointer to current :class:`LightningModule`.
        Return:
            LightningModule with batchnorm layers synchronized between process groups
        """
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=None)

        return model
