## Basic Examples

Use these examples to test how Lightning works.

## MNIST Examples

Here are 5 MNIST examples showing you how to gradually convert from pure PyTorch to PyTorch Lightning.

The transition through [LightningLite](https://pytorch-lightning.readthedocs.io/en/stable/starter/lightning_lite.rst) from pure PyTorch is optional but it might be helpful to learn about it.

#### 1. Image Classifier with Vanilla PyTorch

Trains a simple CNN over MNIST using vanilla PyTorch.

```bash
# CPU
python mnist_examples/image_classifier_1_pytorch.py
```

______________________________________________________________________

#### 2. Image Classifier with LightningLite

This script shows you how to scale the previous script to enable GPU and multi-GPU training using [LightningLite](https://pytorch-lightning.readthedocs.io/en/stable/starter/lightning_lite.html).

```bash
# CPU / multiple GPUs if available
python mnist_examples/image_classifier_2_lite.py
```

______________________________________________________________________

#### 3. Image Classifier - Conversion from Lite to Lightning

This script shows you how to prepare your conversion from [LightningLite](https://pytorch-lightning.readthedocs.io/en/stable/starter/lightning_lite.html) to `LightningModule`.

```bash
# CPU / multiple GPUs if available
python mnist_examples/image_classifier_3_lite_to_lightning_module.py
```

______________________________________________________________________

#### 4. Image Classifier with LightningModule

This script shows you the result of the conversion to the `LightningModule` and finally all the benefits you get from the Lightning ecosystem.

```bash
# CPU
python mnist_examples/image_classifier_4_lightning_module.py

# GPUs (any number)
python mnist_examples/image_classifier_4_lightning_module.py --trainer.accelerator 'gpu' --trainer.devices 2
```

______________________________________________________________________

#### 5. Image Classifier with LightningModule and LightningDataModule

This script shows you how to extract the data related components into a `LightningDataModule`.

```bash
# CPU
python mnist_examples/image_classifier_5_lightning_datamodule.py

# GPUs (any number)
python mnist_examples/image_classifier_5_lightning_datamodule.py --trainer.accelerator 'gpu' --trainer.devices 2

# Distributed Data Parallel (DDP)
python mnist_examples/image_classifier_5_lightning_datamodule.py --trainer.accelerator 'gpu' --trainer.devices 2 --trainer.strategy 'ddp'
```

______________________________________________________________________

#### AutoEncoder

This script shows you how to implement a CNN auto-encoder.

```bash
# CPU
python autoencoder.py

# GPUs (any number)
python autoencoder.py --trainer.accelerator 'gpu' --trainer.devices 2

# Distributed Data Parallel (DDP)
python autoencoder.py --trainer.accelerator 'gpu' --trainer.devices 2 --trainer.strategy 'ddp'
```

______________________________________________________________________

#### Backbone Image Classifier

This script shows you how to implement a `LightningModule` as a system.
A system describes a `LightningModule` which takes a single `torch.nn.Module` which makes exporting to producion simpler.

```bash
# CPU
python backbone_image_classifier.py

# GPUs (any number)
python backbone_image_classifier.py --trainer.accelerator 'gpu' --trainer.devices 2

# Distributed Data Parallel (DDP)
python backbone_image_classifier.py --trainer.accelerator 'gpu' --trainer.devices 2 --trainer.strategy 'ddp'
```

______________________________________________________________________

#### PyTorch Profiler

This script shows you how to activate the [PyTorch Profiler](https://github.com/pytorch/kineto) with Lightning.

```bash
python profiler_example.py
```
