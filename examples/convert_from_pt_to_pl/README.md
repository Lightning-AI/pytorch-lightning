## MNIST Examples

Here are 5 MNIST examples showing you how to gradually convert from pure PyTorch to PyTorch Lightning.

The transition through [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/stable/lightning_lite.rst) from pure PyTorch is optional but it might be helpful to learn about it.

#### 1. Image Classifier with Vanilla PyTorch

Trains a simple CNN over MNIST using vanilla PyTorch.

```bash
# CPU
python image_classifier_1_pytorch.py
```

______________________________________________________________________

#### 2. Image Classifier with LightningLite

This script shows you how to scale the previous script to enable GPU and multi-GPU training using [LightningLite](https://pytorch-lightning.readthedocs.io/en/stable/starter/lightning_lite.html).

```bash
# CPU / multiple GPUs if available
python image_classifier_2_lite.py
```

______________________________________________________________________

#### 3. Image Classifier - Conversion from Lite to Lightning

This script shows you how to prepare your conversion from [LightningLite](https://pytorch-lightning.readthedocs.io/en/stable/starter/lightning_lite.html) to `LightningModule`.

```bash
# CPU / multiple GPUs if available
python image_classifier_3_lite_to_lightning_module.py
```

______________________________________________________________________

#### 4. Image Classifier with LightningModule

This script shows you the result of the conversion to the `LightningModule` and finally all the benefits you get from Lightning.

```bash
# CPU
python image_classifier_4_lightning_module.py

# GPUs (any number)
python image_classifier_4_lightning_module.py --trainer.accelerator 'gpu' --trainer.devices 2
```

______________________________________________________________________

#### 5. Image Classifier with LightningModule and LightningDataModule

This script shows you how to extract the data related components into a `LightningDataModule`.

```bash
# CPU
python image_classifier_5_lightning_datamodule.py

# GPUs (any number)
python image_classifier_5_lightning_datamodule.py --trainer.accelerator 'gpu' --trainer.devices 2

# Distributed Data parallel
python image_classifier_5_lightning_datamodule.py --trainer.accelerator 'gpu' --trainer.devices 2 --trainer.strategy 'ddp'
```
