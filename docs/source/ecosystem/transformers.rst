Lightning Transformers
======================

`Lightning Transformers <https://lightning-transformers.readthedocs.io/en/latest/>`_ offers a flexible interface for training and fine-tuning SOTA Transformer models
using the :doc:`PyTorch Lightning Trainer <../common/trainer>`.

.. code-block:: bash

    pip install lightning-transformers

In Lightning Transformers, we offer the following benefits:

- Powered by `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ - Accelerators, custom Callbacks, Loggers, and high performance scaling with minimal changes.
- Backed by `HuggingFace Transformers <https://huggingface.co/transformers/>`_ models and datasets, spanning multiple modalities and tasks within NLP/Audio and Vision.
- Task Abstraction for Rapid Research & Experimentation - Build your own custom transformer tasks across all modalities with little friction.
- Powerful config composition backed by `Hydra <https://hydra.cc/>`_ - simply swap out models, optimizers, schedulers task, and many more configurations without touching the code.
- Seamless Memory and Speed Optimizations - Out-of-the-box training optimizations such as `DeepSpeed ZeRO <https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#deepspeed>`_ or `FairScale Sharded Training <https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#sharded-training>`_ with no code changes.

-----------------

Using Lightning-Transformers
----------------------------

Lightning Transformers has a collection of tasks for common NLP problems such as `language_modeling <https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/language_modeling.html#language-modeling>`_,
`translation <https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/translation.html#translation>`_ and more. To use, simply:

1. Pick a task to train (passed to ``train.py`` as ``task=``)

2. Pick a dataset (passed to ``train.py`` as ``dataset=``)

3. Customize the backbone, optimizer, or any component within the config

4. Add any :doc:`Lightning supported parameters and optimizations <../common/trainer>`

.. code-block:: bash

    python train.py \
        task=<TASK> \
        dataset=<DATASET>
        backbone.pretrained_model_name_or_path=<BACKBONE> # Optionally change the HF backbone
        optimizer=<OPTIMIZER> # Optionally specify optimizer (Default AdamW)
        trainer.<ANY_TRAINER_FLAGS> # Optionally specify Lightning trainer arguments


To learn more about Lightning Transformers, please refer to the `Lightning Transformers documentation <https://lightning-transformers.readthedocs.io/en/latest/>`_.
