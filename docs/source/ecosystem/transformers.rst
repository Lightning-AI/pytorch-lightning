Lightning Transformers
======================

`Lightning Transformers <https://lightning-transformers.readthedocs.io/en/latest/>`_ offers a flexible interface for training and fine-tuning SOTA Transformer models
using the `PyTorch Lightning Trainer <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_.

.. code-block:: bash

    pip install lightning-transformers

In Lightning Transformers, we offer the following benefits:

- Powered by `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ - Accelerators, custom Callbacks, Loggers, and high performance scaling with minimal changes.
- Backed by `HuggingFace Transformers <https://huggingface.co/transformers/>`_ models and datasets, spanning multiple modalities and tasks within NLP/Audio and Vision.
- Task Abstraction for Rapid Research & Experimentation - Build your own custom transformer tasks across all modalities with little friction.
- Powerful config composition backed by `Hydra <https://hydra.cc/>`_ - simply swap out models, optimizers, schedulers task and many more configurations without touching the code.
- Seamless Memory and Speed Optimizations - Out of the box training optimizations such as `DeepSpeed ZeRO <https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#deepspeed>`_ or `FairScale Sharded Training <https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#sharded-training>`_ with no code changes.

-----------------
