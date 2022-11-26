## MNIST Examples

Here are two MNIST classifiers implemented in PyTorch.
The first one is implemented in pure PyTorch, but isn't easy to scale.
The second one is using [Lightning Lite](https://pytorch-lightning.readthedocs.io/en/stable/starter/lightning_lite.html) to accelerate and scale the model.

#### 1. Image Classifier with Vanilla PyTorch

Trains a simple CNN over MNIST using vanilla PyTorch. It only supports singe GPU training.

```bash
# CPU
python image_classifier_1_pytorch.py
```

______________________________________________________________________

#### 2. Image Classifier with Lightning Lite

This script shows you how to scale the pure PyTorch code to enable GPU and multi-GPU training using [Lightning Lite](https://pytorch-lightning.readthedocs.io/en/stable/starter/lightning_lite.html).

```bash
# CPU
lightning run model image_classifier_2_lite.py

# GPU (CUDA or M1 Mac)
lightning run model image_classifier_2_lite.py --accelerator=gpu

# Multiple GPUs
lightning run model image_classifier_2_lite.py --accelerator=gpu --devices=4
```
