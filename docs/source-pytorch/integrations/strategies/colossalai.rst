:orphan:

###########
Colossal-AI
###########

The `Colossal-AI strategy <https://github.com/Lightning-AI/lightning-colossalai>`_ implements ZeRO-DP with chunk-based memory management.
With this chunk mechanism, really large models can be trained with a small number of GPUs.
It supports larger trainable model size and batch size than usual heterogeneous training by reducing CUDA memory fragments and CPU memory consumption.
Also, it speeds up this kind of heterogeneous training by fully utilizing all kinds of resources.

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

When enabling chunk mechanism, a set of consecutive parameters are stored in a chunk, and then the chunk is sharded across different processes.
This can reduce communication and data transmission frequency and fully utilize communication and PCI-E bandwidth, which makes training faster.

Unlike traditional implementations, which adopt static memory partition, we implemented a dynamic heterogeneous memory management system named Gemini.
During the first training step, the warmup phase will sample the maximum non-model data memory (memory usage expect parameters, gradients, and optimizer states).
In later training, it will use the collected memory usage information to evict chunks dynamically.
Gemini allows you to fit much larger models with limited GPU memory.

According to our benchmark results, we can train models with up to 24 billion parameters in 1 GPU.

You can install the Colossal-AI integration by running

.. code-block:: bash

    pip install lightning-colossalai

This will install both the `colossalai <https://colossalai.org/download>`_ package as well as the ``ColossalAIStrategy`` for the Lightning Trainer:

.. code-block:: python

    trainer = Trainer(strategy="colossalai", precision=16, devices=...)


You can tune several settings by instantiating the strategy objects and pass options in:

.. code-block:: python

    from lightning_colossalai import ColossalAIStrategy

    strategy = ColossalAIStrategy(...)
    trainer = Trainer(strategy=strategy, precision=16, devices=...)


See a full example of a benchmark with the a `GPT-2 model <https://github.com/hpcaitech/ColossalAI-Pytorch-lightning/tree/main/benchmark/gpt>`_ of up to 24 billion parameters

.. note::

    *   The only accelerator which ColossalAI supports is ``"gpu"``. But CPU resources will be used when the placement policy is set to "auto" or "cpu".

    *   The only precision which ColossalAI allows is 16-bit mixed precision (FP16).

    *   It only supports a single optimizer, which must be ``colossalai.nn.optimizer.CPUAdam`` or ``colossalai.nn.optimizer.
        HybridAdam`` now. You can set ``adamw_mode`` to False to use normal Adam. Noticing that ``HybridAdam`` is highly optimized, it uses fused CUDA kernel and parallel CPU kernel.
        It is recomended to use ``HybridAdam``, since it updates parameters in GPU and CPU both.

    *   Your model must be created using the :meth:`~lightning.pytorch.core.module.LightningModule.configure_sharded_model` method.

    *   ``ColossalaiStrategy`` doesn't support gradient accumulation as of now.

.. _colossal_placement_policy:

Model Definition
================

ColossalAI requires the layers of your model to be created in the special :meth:`~lightning.pytorch.core.module.LightningModule.configure_sharded_model` hook.
This allows the strategy to efficiently shard your model before materializing the weight tensors.

.. code-block:: python

    class MyModel(LightningModule):
        def __init__(self):
            super().__init__()
            # don't instantiate layers here
            # move the creation of layers to `configure_sharded_model`

        def configure_sharded_model(self):
            # create all your layers here
            self.layers = nn.Sequential(...)


Placement Policy
================

Placement policies can help users fully exploit their GPU-CPU heterogeneous memory space for better training efficiency.
There are three options for the placement policy.
They are "cpu", "cuda" and "auto" respectively.

When the placement policy is set to "cpu", all participated parameters will be offloaded into CPU memory immediately at the end of every auto-grad operation.
In this way, "cpu" placement policy uses the least CUDA memory.
It is the best choice for users who want to exceptionally enlarge their model size or training batch size.

When using "cuda" option, all parameters are placed in the CUDA memory, no CPU resources will be used during the training.
It is for users who get plenty of CUDA memory.

The third option, "auto", enables Gemini.
It monitors the consumption of CUDA memory during the warmup phase and collects CUDA memory usage of all auto-grad operations.
In later training steps, Gemini automatically manages the data transmission between GPU and CPU according to collected CUDA memory usage information.
It is the fastest option when CUDA memory is enough.

Here's an example of changing the placement policy to "cpu".

.. code-block:: python

    from lightning_colossalai import ColossalAIStrategy

    model = MyModel()
    my_strategy = ColossalAIStrategy(placement_policy="cpu")
    trainer = Trainer(accelerator="gpu", devices=4, precision=16, strategy=my_strategy)
    trainer.fit(model)
