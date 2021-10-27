## MNIST Examples

5 MNIST examples showing how to gradually convert from pure PyTorch to PyTorch Lightning.

The transition through [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.rst) from pure PyTorch is optional but it might be helpful to learn about it.

#### 1 . Image Classifier with Vanilla PyTorch

Trains a simple CNN over MNIST using vanilla PyTorch.

```bash
# cpu
python image_classifier_1_pytorch.py
```

______________________________________________________________________

#### 2. Image Classifier with LightningLite

Trains a simple CNN over MNIST using [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.rst).

```bash
# cpu / multiple gpus if available
python image_classifier_2_lite.py
```

______________________________________________________________________

#### 3. Image Classifier - Conversion Lite to Lightning

Trains a simple CNN over MNIST where `LightningLite` is almost a `LightningModule`.

```bash
# cpu / multiple gpus if available
python image_classifier_3_lite_to_lightning.py
```

______________________________________________________________________

#### 4. Image Classifier with LightningModule

Trains a simple CNN over MNIST with `Lightning Trainer` and the converted `LightningModule`.

```bash
# cpu
python image_classifier_4_lightning.py

# gpus (any number)
python image_classifier_4_lightning.py --trainer.gpus 2
```

______________________________________________________________________

#### 5. Image Classifier with LightningModule + LightningDataModule

Trains a simple CNN over MNIST with `Lightning Trainer` and the converted `LightningModule` and `LightningDataModule`

```bash
# cpu
python image_classifier_5_lightning_datamodule.py

# gpus (any number)
python image_classifier_5_lightning_datamodule.py --trainer.gpus 2

# dataparallel
python image_classifier_5_lightning_datamodule.py --trainer.gpus 2 --trainer.accelerator 'dp'
```
