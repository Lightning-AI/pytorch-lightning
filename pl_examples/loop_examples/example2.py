import os
from torch.utils.data import DataLoader

from pl_examples.bug_report_model import RandomDataset
from pl_examples.loop_examples.example1 import BoringModel
from pl_examples.loop_examples.simple_loop import SimpleLoop
from pytorch_lightning import Trainer


def run():
    """
    This example shows how to replace the FitLoop on the Trainer with a very simple, custom iteration-based
    training loop.
    """
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        weights_summary=None,
        progress_bar_refresh_rate=1,
    )

    # instantiate the new loop
    simple_loop = SimpleLoop(num_iterations=1000)

    # replace the fit loop
    # the trainer reference will be set internally
    trainer.fit_loop = simple_loop

    # fit using the new loop!
    trainer.fit(model, train_dataloader=train_data)


if __name__ == '__main__':
    run()
