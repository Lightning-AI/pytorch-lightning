# Examples

*Note that some examples may rely on new features that are only available in the development branch and may be incompatible with any releases.*
*If you see any errors, you might want to consider switching to a version tag you would like to run examples with.*
*For example, if you're using `pytorch-lightning==1.6.4` in your environment and seeing issues, run examples of the tag [1.6.4](https://github.com/Lightning-AI/lightning/tree/1.6.4/pl_examples).*

______________________________________________________________________

## Lightning Fabric Examples

We show how to accelerate your PyTorch code with [Lightning Fabric](https://lightning.ai/docs/fabric) with minimal code changes.
You stay in full control of the training loop.

- [MNIST: Vanilla PyTorch vs. Fabric](fabric/image_classifier/README.md)
- [DCGAN: Vanilla PyTorch vs. Fabric](fabric/dcgan/README.md)

______________________________________________________________________

## Lightning Trainer Examples

In this folder, we have 2 simple examples that showcase the power of the Lightning Trainer.

- [Image Classifier](pytorch/basics/backbone_image_classifier.py) (trains arbitrary datasets with arbitrary backbones).
- [Autoencoder](pytorch/basics/autoencoder.py)

______________________________________________________________________

## Domain Examples

This folder contains older examples. You should instead use the examples
in [Lightning Bolts](https://lightning.ai/docs/pytorch/stable/ecosystem/bolts.html)
for advanced use cases.
