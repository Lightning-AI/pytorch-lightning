import os
from torch.utils.data import DataLoader

from pl_examples.bug_report_model import RandomDataset, BoringModel
from pytorch_lightning import Trainer
from pytorch_lightning.loops import EvaluationLoop, TrainingBatchLoop


def run():
    """
    This example shows how to switch out an individual loop.
    Here, we want to take the default FitLoop from Lightning but switch out

    1. the batch_loop inside the training epoch loop
    2. the val_loop inside the training epoch loop

    """
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()

    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        weights_summary=None,
    )

    # instantiate the new batch- and validation loop
    new_batch_loop = TrainingBatchLoop()
    new_val_loop = EvaluationLoop()

    # call connect on the existing, default fit_loop.epoch_loop
    trainer.fit_loop.epoch_loop.connect(batch_loop=new_batch_loop, val_loop=new_val_loop)

    # the new batch loop is registered and the trainer got linked internally
    assert trainer.fit_loop.epoch_loop.batch_loop == new_batch_loop
    assert trainer.fit_loop.epoch_loop.batch_loop.trainer == trainer

    # this uses the new custom batch loop
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)


if __name__ == '__main__':
    run()
