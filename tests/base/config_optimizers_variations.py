from torch import optim


class ConfigureOptimizersVariationsMixin:
    def configure_optimizers(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        # try no scheduler for this model (testing purposes)
        if self.hparams.optimizer_name == 'lbfgs':
            optimizer = optim.LBFGS(self.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
