import tests.base.model_utils as mutils


class TrainDataloaderVariationsMixin:

    def train_dataloader(self):
        return mutils.dataloader(
            train=True,
            data_root=self.hparams.data_root,
            batch_size=self.hparams.batch_size,
        )
