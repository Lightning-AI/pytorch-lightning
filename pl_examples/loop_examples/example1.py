import os

from torch.utils.data import DataLoader

from pl_examples.bug_report_model import BoringModel, RandomDataset
from pytorch_lightning import Trainer
from pytorch_lightning.loops import EvaluationLoop, FitLoop, TrainingBatchLoop, TrainingEpochLoop


def run():
    """
    This example demonstrates how loops are linked together.
    Here we form a simple tree structure of three basic loops that make up the FitLoop:

    - Trainer
        - fit_loop: FitLoop
            - epoch_loop: TrainingEpochLoop
                - batch_loop: TrainingBatchLoop
                - val_loop: EvaluationLoop
    """
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()

    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        weights_summary=None,
    )

    # construct loops
    fit_loop = FitLoop(max_epochs=2)
    train_epoch_loop = TrainingEpochLoop(min_steps=0, max_steps=2)
    train_batch_loop = TrainingBatchLoop()
    val_loop = EvaluationLoop()

    # connect loops together
    train_epoch_loop.connect(batch_loop=train_batch_loop, val_loop=val_loop)
    fit_loop.connect(epoch_loop=train_epoch_loop)

    # connect fit loop to trainer (main entry point for the call in trainer.fit())
    trainer.fit_loop = fit_loop

    # this will use the newly constructed loop!
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)

    # this will still use the default test loop
    trainer.test(model, dataloaders=test_data)


if __name__ == '__main__':
    run()
