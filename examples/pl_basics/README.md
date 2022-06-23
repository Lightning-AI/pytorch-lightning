## Basic Examples

Use these examples to test how Lightning works.

### AutoEncoder

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

### Backbone Image Classifier

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

### PyTorch Profiler

This script shows you how to activate the [PyTorch Profiler](https://github.com/pytorch/kineto) with Lightning.

```bash
python profiler_example.py
```
