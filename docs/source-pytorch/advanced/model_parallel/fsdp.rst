:orphan:

.. _fully-sharded-training:

###################################################
Train models with billions of parameters using FSDP
###################################################

Use Fully Sharded Data Parallel (FSDP) to train large models with billions of parameters efficiently on multiple GPUs and across multiple machines.

Today, large models with billions of parameters are trained with many GPUs across several machines in parallel.
Even a single H100 GPU with 80 GB of VRAM (one of the biggest today) is not enough to train just a 30B parameter model (even with batch size 1 and 16-bit precision).
The memory consumption for training is generally made up of

1. the model parameters,
2. the layer activations (forward),
3. the gradients (backward) and
4. the optimizer states (e.g., Adam has two additional exponential averages per parameter).

|

When the sum of these memory components exceed the VRAM of a single GPU, regular data-parallel training (DDP) can no longer be employed.
One of the methods that can alleviate this limitation is called **Fully Sharded Data Parallel (FSDP)**, and in this guide, you will learn how to effectively scale large models with it.


----


***************************
Checklist: When to use FSDP
***************************

|

✅   I have multiple GPUs

✅   I have tried regular DDP training with batch size 1 but I run out of memory

✅   I have PyTorch 2.0 or newer installed


----


**********************
Enable FSDP in Trainer
**********************


To enable model-parallel training with FSDP in a single-line change, set ``strategy="fsdp"``:

.. code-block:: python

    trainer = L.Trainer(accelerator="cuda", devices=2, strategy="fsdp")

As we will see in the next sections, there are many settings we can tune to optimize memory usage and throughput, scaling to massively large models.
This is equivalent to the above, but will let us configure additional settings later:

.. code-block:: python

    from lightning.pytorch.strategies import FSDPStrategy

    trainer = L.Trainer(accelerator="cuda", devices=2, strategy=FSDPStrategy())


Here is a full code example:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    import lightning as L
    from lightning.pytorch.strategies import FSDPStrategy
    from lightning.pytorch.demos import Transformer, WikiText2


    class LanguageModel(L.LightningModule):
        def __init__(self, vocab_size):
            super().__init__()
            self.model = Transformer(  # 1B parameters
                vocab_size=vocab_size,
                nlayers=32,
                nhid=4096,
                ninp=1024,
                nhead=64,
            )

        def training_step(self, batch):
            input, target = batch
            output = self.model(input, target)
            loss = F.nll_loss(output, target.view(-1))
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.1)


    L.seed_everything(42)

    # Data
    dataset = WikiText2()
    train_dataloader = DataLoader(dataset)

    # Model
    model = LanguageModel(vocab_size=dataset.vocab_size)

    # Trainer
    trainer = L.Trainer(accelerator="cuda", devices=2, strategy=FSDPStrategy())
    trainer.fit(model, train_dataloader)
    trainer.print(torch.cuda.memory_summary())



We will reuse this Transformer example throughout the guide, optimize speed and memory usage, and compare it to regular DDP training.


----


*********************
Identify large layers
*********************

Models that have many large layers like linear layers in LLMs, ViTs, etc. with >100M parameters will benefit the most from FSDP because the memory they consume through parameters, activations and corresponding optimizer states can be evenly split across all GPUs.
However, one should avoid splitting small layers that have a few thousand parameters because communication overhead would dominate and slow the training down.
We can specify a list of layer classes in the **wrapping policy** to inform FSDP which parameters it should wrap:

.. code-block:: python

    # 1. Define a set of layers that FSDP should manage
    #    Here we are choosing the large encoder and decoder layers
    policy = {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}

    # 2. Pass the policy to the FSDPStrategy object
    strategy = FSDPStrategy(auto_wrap_policy=policy)

    trainer = L.Trainer(..., strategy=strategy)

.. collapse:: Alternative ways to define the policy (Lightning < 2.1)

    The ``auto_wrap_policy`` argument also accepts the old-style function-policies. For example:

    .. code-block:: python

        from functools import partial

        # 1. Import a suiting wrapping policy from PyTorch
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        # 2. Configure the policy
        policy = partial(size_based_auto_wrap_policy, min_num_params=10000)

        # 3. Pass it to the FSDPStrategy object
        strategy = FSDPStrategy(auto_wrap_policy=policy)

    PyTorch provides several of these functional policies under ``torch.distributed.fsdp.wrap``.

|

Verify that FSDP works with your model by comparing the peak memory usage printed in the CUDA memory summary (see example above) with regular DDP training.
You should see a decrease in allocated memory and a slight increase in iteration time:

.. list-table:: Numbers were produced with A100 40GB GPUs, Lightning 2.1 and PyTorch 2.1.
   :widths: 25 25 25
   :header-rows: 1

   * -
     - DDP
     - FSDP
   * - Memory (MB)
     - 23’125
     - 9’627
   * - Iterations per second
     - 4.31
     - 3.19

----


*****************************
Speed up model initialization
*****************************

The standard practice in PyTorch is to put all model parameters into CPU memory first and then in a second step move them to the GPU device.
However, the larger the model the longer these two steps take.
If you create the large model layers inside the :meth:`~lightning.pytorch.core.hooks.ModelHooks.configure_model` hook, you can initialize very large models quickly and reduce memory peaks.

Before:

.. code-block:: python

    # Slow: Places the model on CPU first
    class LanguageModel(L.LightningModule):
        def __init__(self, vocab_size):
            super().__init__()
            # 1B parameters
            self.model = Transformer(vocab_size=vocab_size, nlayers=32, nhid=4096, ninp=1024, nhead=64)

After:

.. code-block:: python

    # Fast: Delays the model creation until Trainer can place it on GPU
    class LanguageModel(L.LightningModule):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.model = None

        def configure_model(self):
            if self.model is not None:
                return
            self.model = Transformer(  # 1B parameters
                vocab_size=self.vocab_size,
                nlayers=32,
                nhid=4096,
                ninp=1024,
                nhead=64,
            )


It is best practice to make the code in :meth:`~lightning.pytorch.core.hooks.ModelHooks.configure_model` idempotent as shown here.
Learn more about :doc:`efficient initialization of models in Lightning <../model_init>`.


----


******************************
Optimize the sharding strategy
******************************

By default, FSDP will automatically shard 1) the model weights 2) the gradients during backward and 3) the optimizer states across all GPUs of the corresponding layers selected by the auto-wrap-policy.
You can configure the following options to trade-off memory for speed:

.. code-block:: python

    strategy = FSDPStrategy(
        # Default: Shard weights, gradients, optimizer state (1 + 2 + 3)
        sharding_strategy="FULL_SHARD",
        # Shard gradients, optimizer state (2 + 3)
        sharding_strategy="SHARD_GRAD_OP",
        # Full-shard within a machine, replicate across machines
        sharding_strategy="HYBRID_SHARD",
        # Don't shard anything (similar to DDP)
        sharding_strategy="NO_SHARD",
    )
    trainer = L.Trainer(..., strategy=strategy)


**Recipe for choosing a sharding strategy:**

1. Try the default settings first (FULL_SHARD). This is the slowest but will save you the most memory.
2. Try SHARD_GRAD_OP. If you run out of memory, revert back to the default (FULL_SHARD). Otherwise you should expect to see an increase in iteration speed.
3. If you are training across many machines, try HYBRID_SHARD.

|

Here is the memory and speed impact for each option when configured in our example code:

.. list-table:: Numbers were produced with A100 40GB GPUs, Lightning 2.1 and PyTorch 2.1.
   :widths: 25 25 25 25 25
   :header-rows: 1

   * -
     - DDP
     - NO_SHARD
     - SHARD_GRAD_OP
     - FULL_SHARD
   * - Memory (MB)
     - 23’125
     - 19’296
     - 11’772
     - 9’627
   * - Iterations per second
     - 4.31
     - 3.04
     - 3.61
     - 3.19


----


**************************
Trade-off speed for memory
**************************

If you are short on GPU memory because you are training large models with 10+ billion parameters or require extreme batch sizes, consider trading off speed for more memory by enabling activation checkpointing or CPU offload.


Activation checkpointing
========================

Activations, the intermediate outputs of layers, are stored during the forward pass and needed during the backward pass to compute the gradients.
By enabling activation checkpointing, we can choose to discard and recompute selected layer activations dynamically during the backward pass when they are required, instead of storing them throughout the forward pass.
While this approach may slightly reduce training speed, it significantly reduces memory consumption.
The freed-up memory can then be allocated to increase the model's capacity or accommodate larger batch sizes, resulting in potential performance improvements.

To enable activation checkpointing, pass in the list of layers to checkpoint.
This is typically your transformer block (including attention + feed-forward):

.. code-block:: python

    strategy = FSDPStrategy(
        # Enable activation checkpointing on these layers
        activation_checkpointing_policy={
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
        },
    )
    trainer = L.Trainer(..., strategy=strategy)


As in our example, it is typical to set the ``activation_checkpointing_policy`` the same as ``auto_wrap_policy``.


Offload parameters to CPU
=========================

The most drastic GPU memory savings can be achieved by offloading parameters to the CPU:

.. code-block:: python

    # Set `cpu_offload=True`
    strategy = FSDPStrategy(..., cpu_offload=True)
    trainer = L.Trainer(..., strategy=strategy)

The drawback is a much slower training speed due to the added communication between CPU and GPU for transferring parameters in every forward pass.
You should use this only if you have enough CPU memory and other scaling methods don’t give you enough memory savings.
In our example, we see a 3.5x memory saving, but a significant increase in iteration time:

.. list-table:: Numbers were produced with A100 40GB GPUs, Lightning 2.1 and PyTorch 2.1.
   :widths: 25 25 25 25
   :header-rows: 1

   * -
     - DDP
     - FSDP
     - FSDP + CPU offload
   * - Memory (MB)
     - 23’125
     - 9’627
     - 2’790
   * - Iterations per second
     - 4.31
     - 3.19
     - 0.02


----


*****************
Save a checkpoint
*****************

Since training large models can be very expensive, it is best practice to checkpoint the training state periodically in case it gets interrupted unexpectedly.
Lightning saves a checkpoint every epoch by default, and there are :ref:`several settings to configure the checkpointing behavior in detail <checkpointing>`.

.. code-block:: python

    # Default: Saves a checkpoint every epoch
    trainer = L.Trainer()
    trainer.fit(model)

    # You can also manually trigger a checkpoint at any time
    trainer.save_checkpoint("path/to/checkpoint/file")

    # DON'T do this (inefficient):
    # torch.save("path/to/checkpoint/file", model.state_dict())

For single-machine training this typically works fine, but for larger models saving a checkpoint can become slow (minutes not seconds) or overflow CPU memory (OOM) depending on the system.
To reduce memory peaks and speed up the saving to disk, set ``state_dict_type="sharded"``:

.. code-block:: python

    # Default: Save a single, consolidated checkpoint file
    strategy = FSDPStrategy(state_dict_type="full")

    # Save individual files with state from each process
    strategy = FSDPStrategy(state_dict_type="sharded")


With this, each process/GPU will save its own file into a folder at the given path by default.
The resulting checkpoint folder will have this structure:

.. code-block:: text

    path/to/checkpoint/file
    ├── .metadata
    ├── __0_0.distcp
    ├── __1_0.distcp
    ...
    └── meta.pt

The “sharded” checkpoint format is the most efficient to save and load in Lightning.

**Which checkpoint format should I use?**

- ``state_dict_type="sharded"``: Use for pre-training very large models. It is fast and uses less memory, but it is less portable. An extra step is needed to :doc:`convert the sharded checkpoint into a regular checkpoint file <../../common/checkpointing_expert>`.
- ``state_dict_type="full"``: Use when pre-training small to moderately large models (less than 10B parameters), when fine-tuning, and when portability is required.


----


*****************
Load a checkpoint
*****************

You can easily :ref:`load checkpoints <checkpointing>` saved by Lightning to resume training:

.. code-block:: python

    trainer = L.Trainer(...)

    # Restore the training progress, weights, and optimizer state
    trainer.fit(model, ckpt_path="path/to/checkpoint/file")


The Trainer will automatically recognize whether the provided path contains a checkpoint saved with ``state_dict_type="full"`` or ``state_dict_type="sharded"``.
Checkpoints saved with ``state_dict_type="full"`` can be loaded by all strategies, but sharded checkpoints can only be loaded by FSDP.
Read :ref:`the checkpoints guide <checkpointing>` to explore more features.


----


**********************************
Advanced performance optimizations
**********************************

If you’ve reached a good understanding of how the different FSDP settings impact the memory usage and speed of your model, here are a few more to squeeze out the last bit of performance.
These settings really depend on the specific use cases, so you will have to turn them on and off to see the impact on your model.


Disable foreach in the optimizer
================================

The commonly used optimizers in PyTorch have a setting ``foreach=True|False`` that speeds up the parameter and state updates when enabled.
However, you might see a slight memory peak and the larger the model is, the more noticeable it can be.
Consider disabling the ``foreach`` option if undesired memory patterns occur:

.. code-block:: python

    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)

`See the full list of optimizers that support this <https://pytorch.org/docs/stable/optim.html#algorithms>`_.


Limit all-gathers
=================

If you are running training close to the max.
GPU memory limit, you might be getting so-called CUDA malloc retries.
This is essentially the GPU running out of memory but before crashing completely, it tries to find some unused or cached memory it can free.
When they happen frequently, these retries can have a significant impact on speed.
Normally, you would decrease the batch size slightly to avoid it.
With FSDP, you have one more knob you can tweak to combat the issue, by setting ``limit_all_gathers=True``:

.. code-block:: python

    strategy = FSDPStrategy(
        # Default: The CPU will schedule the transfer of weights between GPUs
        # at will, sometimes too aggressively
        limit_all_gathers=False,
        # Enable this if you are close to the max. GPU memory usage
        limit_all_gathers=True,
    )
    trainer = L.Trainer(..., strategy=strategy)

You can monitor CUDA malloc retries in the output of ``torch.cuda.memory_summary()`` for example, or through the PyTorch profiler.


Manual wrapping
===============

Manual wrapping can be useful to explore complex sharding strategies by applying ``wrap`` selectively to some parts of the model.
To activate parameter sharding with manual wrapping, you can wrap your model using the ``wrap`` function.
Internally in Lightning, we enable a context manager around the :meth:`~lightning.pytorch.core.hooks.ModelHooks.configure_model` hook to make sure the ``wrap`` parameters are passed correctly.

Here is an example that uses ``wrap`` to create a model:

.. code-block:: python

    import torch
    import torch.nn as nn
    import lightning as L

    from torch.distributed.fsdp.wrap import wrap


    class MyModel(L.LightningModule):
        def configure_model(self):
            self.linear_layer = nn.Linear(32, 32)
            self.block = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32))

            # Modules get sharded across processes as soon as they are wrapped with `wrap`.
            linear_layer = wrap(self.linear_layer)

            for i, layer in enumerate(self.block):
                self.block[i] = wrap(layer)

            self.model = nn.Sequential(linear_layer, nn.ReLU(), self.block)

        def configure_optimizers(self):
            return torch.optim.AdamW(self.model.parameters())


    model = MyModel()
    trainer = L.Trainer(accelerator="cuda", devices=4, strategy="fsdp", precision=16)
    trainer.fit(model)

When not using FSDP, these ``wrap`` calls are a no-op.
This means once the changes have been made, there is no need to remove the changes for other strategies.
In this case, Lightning will not re-wrap your model, so you don't need to set ``FSDPStrategy(auto_wrap_policy=...)``.
Check out `this tutorial <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__ to learn more about it.
