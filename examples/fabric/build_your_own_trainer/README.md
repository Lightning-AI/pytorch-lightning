## Build Your Own Trainer (BYOT)

This example demonstrates how easy it is to build a fully customizable trainer for your `LightningModule` using `Fabric`.
It is built upon `lightning.fabric` for hardware and training orchestration and consists of two files:

- trainer.py contains the actual `MyCustomTrainer` implementation
- run.py contains a script utilizing this trainer for training a very simple MNIST module.

### Run

To run this example, call `python run.py`

### Requirements

This example has the following requirements which need to be installed on your python environment:

- `lightning`
- `torchmetrics`
- `torch`
- `torchvision`
- `tqdm`

to install them with the appropriate versions run:

```bash
pip install "lightning>=2.0" "torchmetrics>=0.11" "torchvision>=0.14" "torch>=1.13" tqdm
```
