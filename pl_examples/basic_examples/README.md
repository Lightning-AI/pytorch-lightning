## Basic Examples

Use these examples to test how Lightning works.

## MNIST Examples

5 MNIST examples showing how to gradually convert from pure PyTorch to PyTorch Lightning.

The transition through [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.rst) from pure PyTorch is optional but it might be helpful to learn about it.

#### 1 . Image Classifier with Vanilla PyTorch

Trains a simple CNN over MNIST using vanilla PyTorch.

```bash
# cpu
python mnist_examples/image_classifier_1_pytorch.py
```

______________________________________________________________________

#### 2. Image Classifier with LightningLite

This script shows you how to scale the previous script to enable GPU and multi GPU training using [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.rst).

```bash
# cpu / multiple gpus if available
python mnist_examples/image_classifier_2_lite.py
```

______________________________________________________________________

#### 3. Image Classifier - Conversion Lite to Lightning

This script shows you to prepare your conversion from [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.rst)
to `LightningModule`.

```bash
# cpu / multiple gpus if available
python mnist_examples/image_classifier_3_lite_to_lightning.py
```

______________________________________________________________________

#### 4. Image Classifier with LightningModule

This script shows you how the result of the conversion to the `LightningModule` and finally get all the benefits from Lightning.

```bash
# cpu
python mnist_examples/image_classifier_4_lightning.py

# gpus (any number)
python mnist_examples/image_classifier_4_lightning.py --trainer.gpus 2
```

______________________________________________________________________

#### 5. Image Classifier with LightningModule + LightningDataModule

This script shows you how extracts the data related components to a `LightningDataModule`.

```bash
# cpu
python mnist_examples/image_classifier_5_lightning_datamodule.py

# gpus (any number)
python mnist_examples/image_classifier_5_lightning_datamodule.py --trainer.gpus 2

# data parallel
python mnist_examples/image_classifier_5_lightning_datamodule.py --trainer.gpus 2 --trainer.accelerator 'dp'
```

______________________________________________________________________

#### Autoencoder

Showing the power of a system... arbitrarily complex training loops

```bash
# cpu
python autoencoder.py

# gpus (any number)
python autoencoder.py --trainer.gpus 2

# Distributed Data Parallel
python autoencoder.py --trainer.gpus 2 --trainer.accelerator ddp
```
