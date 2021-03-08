import os
from numpy.lib.npyio import save

import torch
from torch.random import seed
from torch.utils.data import Dataset

from pl_examples import cli_lightning_logo
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger

from pytorch_lightning import metrics
from pytorch_lightning.metrics.utils import to_categorical


class RandomDataset(Dataset):
    """
    >>> RandomDataset(size=10, length=20)  # doctest: +ELLIPSIS
    <...bug_report_model.RandomDataset object at ...>
    """

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    """
    >>> BoringModel()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    BoringModel(
      (layer): Linear(...)
    )
    """

    def __init__(self):
        """
        Testing PL Module

        Use as follows:
        - subclass
        - modify the behavior for what you want

        class TestModel(BaseTestModel):
            def training_step(...):
                # do your own thing

        or:

        model = BaseTestModel()
        model.training_epoch_end = None

        """
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self.layer(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x['x'] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


#  NOTE: If you are using a cmd line to run your script,
#  provide the cmd line as below.
#  opt = "--max_epochs 1 --limit_train_batches 1".split(" ")
#  parser = ArgumentParser()
#  args = parser.parse_args(opt)

def test_run():

    class TestModel(BoringModel):

        def __init__(self):
            super().__init__()

            self.valid_acc = metrics.classification.Accuracy(
            )

            self.valid_precision = metrics.classification.Precision(num_classes=1,
                                                                    is_multiclass=False
            )
            self.valid_recall = metrics.classification.Recall(num_classes=1,
                                                              is_multiclass=False
            )

            self.valid_statscores = metrics.classification.StatScores(
                num_classes=1,
                is_multiclass=False,
            )

            self._pl_metrics = [
                "valid_acc",
                "valid_precision",
                "valid_recall"
            ]

            self._compare_metrics = [
                "_accurcay",
                "_precision",
                "_recall"
            ]

        @staticmethod
        def _accurcay(tp, fp, tn, fn):
            return (tp + tn) / (tp + fp + tn + fn)

        @staticmethod
        def _precision(tp, fp, tn, fn):
            return tp / (tp + fp)

        @staticmethod
        def _recall(tp, fp, tn, fn):
            return tp / (tp + fn)

        @staticmethod
        def _y_to_cat(logits, targets):
            y_pred = to_categorical(
                tensor=torch.sigmoid(logits),
                argmax_dim=1
            )
            return y_pred, targets

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)

            bs = torch.tensor([len(output)], dtype=torch.int16).type_as(output)

            # update metrics
            _preds, _targets = self._y_to_cat(
                logits=output,
                targets=torch.multinomial(
                    torch.tensor([.5, .5]),
                    len(output),
                    replacement=True
                )
            )

            for _metname in self._pl_metrics:
                eval("self." + _metname)(
                    preds=_preds,
                    target=_targets
                )

            self.valid_statscores(preds=_preds, target=_targets)

            return {"loss": loss, "batch_size": bs}

        def validation_epoch_end(self, outputs):
            # concat batch sizes
            batch_sizes = torch.stack(
                [x["batch_size"] for x in outputs]
            ).type_as(outputs[0]["loss"])

            # concat losses
            losses = torch.stack(
                [x["loss"] for x in outputs]
            ).type_as(outputs[0]["loss"])

            # calculating weighted mean loss
            avg_loss = torch.sum(losses * batch_sizes) / torch.sum(batch_sizes)

            self.log(
                name="loss/valid",
                value=avg_loss,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True
            )

            # compute metrics
            for _metname in self._pl_metrics:
                self.log(
                    name="pl/" + _metname,
                    value=eval("self." + _metname + ".compute()"),
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True
                )
                eval("self." + _metname + ".reset()")

            tp, fp, tn, fn, _ = self.valid_statscores.compute()
            self.valid_statscores.reset()

            for _metname in self._compare_metrics:
                self.log(
                    name="comparison/" + _metname,
                    value=eval("self." + _metname + "(tp, fp, tn, fn)"),
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True
                )

            for _m in ["tp", "fp", "tn", "fn"]:
                self.log(
                    name="comparison/" + _m,
                    value=eval(_m),
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True
                )

    # fake data
    train_data = torch.utils.data.DataLoader(RandomDataset(32, 100), batch_size=2)
    val_data = torch.utils.data.DataLoader(RandomDataset(32, 100), batch_size=2)
    test_data = torch.utils.data.DataLoader(RandomDataset(32, 100), batch_size=2)

    # logger
    csvlogger = CSVLogger(
        save_dir=os.getcwd()
    )
    # model
    seed_everything(0)
    model = TestModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=5,
        weights_summary=None,
        logger=csvlogger,
        deterministic=True
    )
    trainer.fit(model, train_data, val_data)
    trainer.test(test_dataloaders=test_data)


if __name__ == '__main__':
    cli_lightning_logo()
    test_run()