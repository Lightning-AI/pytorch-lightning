## MNIST Examples

Here are two MNIST classifiers implemented in PyTorch.
The first one is implemented in pure PyTorch, but isn't easy to scale.
The second one is using [Lightning Fabric](https://lightning.ai/docs/fabric) to accelerate and scale the model.

Tip: You can easily inspect the difference between the two files with:

```bash
sdiff train_torch.py train_fabric.py
```

#### 1. Image Classifier with Vanilla PyTorch

Trains a simple CNN over MNIST using vanilla PyTorch. It only supports single GPU training.

```bash
# CPU
python train_torch.py
```

______________________________________________________________________

#### 2. Image Classifier with Lightning Fabric

This script shows you how to scale the pure PyTorch code to enable GPU and multi-GPU training using [Lightning Fabric](https://lightning.ai/docs/fabric).

```bash
# CPU
fabric run train_fabric.py

# GPU (CUDA or M1 Mac)
fabric run train_fabric.py --accelerator=gpu

# Multiple GPUs
fabric run train_fabric.py --accelerator=gpu --devices=4
```
