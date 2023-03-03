## Build Your Own Trainer (BYOT)

This example demonstrates, how easy it is, to build a fully customizable trainer working with a `LightningModule`.
It is built upon `lightning.fabric` for hardware and training orchestration and consists of two files:

- trainer.py contains the actual `FabricTrainer` implementation
- run.py contains a script utilizing this trainer for training a very simple MNIST Module.

### Run

To run this example, call `python run.py`

### Requirements

This example has the following requirements which need to be installed on your python interpreter:

- `lightning`
- `torchmetrics`
- `torch`
- `torchvision`
- `tqdm`
