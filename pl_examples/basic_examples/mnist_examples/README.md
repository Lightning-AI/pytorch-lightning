## MNIST Examples

The following examples contain 5 MNIST examples showing how to gradually convert from pure PyTorch to PyTorch Lightning.

#### 1 . Image Classifier with PyTorch

Trains a simple CNN over MNIST using raw PyTorch.

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

#### 3. Image Classifier - Conversion Lite to Lightning.

Trains a simple CNN over MNIST with a `LightningModule` and `LightningLite`.

```bash
# cpu / multiple gpus if available
python image_classifier_3_lite_to_lightning.py
```

______________________________________________________________________

#### 4. Image Classifier - Conversion Lite to Lightning + Lightning Loops

Trains a simple CNN over MNIST with a `LightningModule` and `LightningLite` and `Loops`.

```bash
# cpu / multiple gpus if available
python image_classifier_4_lite_to_lightning_and_loops.py
```

______________________________________________________________________

#### 5. Image Classifier with Lightning.

Trains a simple CNN over MNIST with a `Trainer` and `LightningModule`.

```bash
# cpu
python image_classifier_5_lightning.py

# gpus (any number)
python image_classifier_5_lightning.py --trainer.gpus 2

# dataparallel
python image_classifier_5_lightning.py --trainer.gpus 2 --trainer.accelerator 'dp'
```
