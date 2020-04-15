class TestDataloaderVariationsMixin:

    def test_dataloader(self):
        return self.dataloader(
            train=False,
            data_root=self.hparams.data_root,
            batch_size=self.hparams.batch_size,
        )
