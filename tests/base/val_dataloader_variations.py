import tests.base.model_utils as mutils


class ValDataloaderVariationsMixin:

    def val_dataloader(self):
        return mutils.dataloader(
            train=False,
            data_root=self.hparams.data_root,
            batch_size=self.hparams.batch_size,
        )
