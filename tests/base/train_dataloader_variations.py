class TrainDataloaderVariationsMixin:

    def train_dataloader(self):
        return self.dataloader(
            train=True,
            data_root=self.hparams.data_root,
            batch_size=self.hparams.batch_size,
        )
