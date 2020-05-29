.. PyTorch-Lightning documentation master file, created by
   sphinx-quickstart on Fri Nov 15 07:48:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyTorch Lightning Documentation
===============================

.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Start Here

   new-project
   introduction_guide

.. toctree::
   :maxdepth: 2
   :name: docs
   :caption: Python API

   callbacks
   hooks
   lightning-module
   loggers
   metrics
   trainer

.. toctree::
   :maxdepth: 1
   :name: Community Examples
   :caption: Community Examples

   Contextual Emotion Detection (DoubleDistilBert) <https://github.com/PyTorchLightning/emotion_transformer>
   FasterRCNN object detection + Hydra <https://github.com/PyTorchLightning/wheat>
   Generative Adversarial Network <https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=TyYOdg8g77P0>
   Hyperparameter optimization with Optuna <https://github.com/optuna/optuna/blob/master/examples/pytorch_lightning_simple.py>
   Image Inpainting using Partial Convolutions <https://github.com/ryanwongsa/Image-Inpainting>
   MNIST on TPU <https://colab.research.google.com/drive/1-_LKx4HwAxl5M6xPJmqAAu444LTDQoa3#scrollTo=BHBz1_AnamN_>
   NER (transformers, TPU) <https://colab.research.google.com/drive/1dBN-wwYUngLYVt985wGs_OKPlK_ANB9D>
   NeuralTexture (CVPR) <https://github.com/PyTorchLightning/neuraltexture>
   Recurrent Attentive Neural Process <https://github.com/PyTorchLightning/attentive-neural-processes>
   Siamese Nets for One-shot Image Recognition <https://github.com/PyTorchLightning/Siamese-Neural-Networks>
   Speech Transformers <https://github.com/PyTorchLightning/speech-transformer-pytorch_lightning>
   Transformers transfer learning (Huggingface) <https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=yr7eaxkF-djf>
   Transformers text classification <https://github.com/ricardorei/lightning-text-classification>
   VAE Library of over 18+ VAE flavors <https://github.com/AntixK/PyTorch-VAE>

.. toctree::
   :maxdepth: 1
   :name: Tutorials
   :caption: Tutorials

   From PyTorch to PyTorch Lightning <https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09>

.. toctree::
   :maxdepth: 1
   :name: project structure
   :caption: Recommended Lightning Project Layout

   Lightning project seed <https://github.com/PyTorchLightning/pytorch-lightning-conference-seed>

.. toctree::
   :maxdepth: 1
   :name: Common Use Cases
   :caption: Common Use Cases

   apex
   slurm
   child_modules
   debugging
   experiment_logging
   experiment_reporting
   early_stopping
   fast_training
   hooks
   hyperparameters
   lr_finder
   multi_gpu
   multiple_loaders
   weights_loading
   optimizers
   profiler
   single_gpu
   sequences
   training_tricks
   transfer_learning
   tpu
   test_set

.. toctree::
   :maxdepth: 1
   :name: community
   :caption: Community


   CODE_OF_CONDUCT.md
   CONTRIBUTING.md
   BECOMING_A_CORE_CONTRIBUTOR.md
   PULL_REQUEST_TEMPLATE.md
   governance.md

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. This is here to make sphinx aware of the modules but not throw an error/warning
.. toctree::
   :hidden:

   api/pytorch_lightning.core
   api/pytorch_lightning.callbacks
   api/pytorch_lightning.loggers
   api/pytorch_lightning.metrics
   api/pytorch_lightning.overrides
   api/pytorch_lightning.profiler
   api/pytorch_lightning.trainer
   api/pytorch_lightning.utilities
