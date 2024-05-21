#####################################################
Training models with billions of parameters with FSDP
#####################################################

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


*********************
Enable FSDP in Fabric
*********************


To enable model-parallel training with FSDP in a single-line change, set ``strategy="fsdp"``:

.. code-block:: python

    fabric = L.Fabric(accelerator="cuda", devices=2, strategy="fsdp")

As we will see in the next sections, there are many settings we can tune to optimize memory usage and throughput, scaling to massively large models.
This is equivalent to the above, but will let us configure additional settings later:

.. code-block:: python

    from lightning.fabric.strategies import FSDPStrategy

    fabric = L.Fabric(accelerator="cuda", devices=2, strategy=FSDPStrategy())


Here is a full code example:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import lightning as L
    from lightning.fabric.strategies import FSDPStrategy
    from lightning.pytorch.demos import Transformer, WikiText2

    fabric = L.Fabric(accelerator="cuda", devices=2, strategy=FSDPStrategy())
    fabric.launch()

    fabric.seed_everything(42)

    with fabric.rank_zero_first():
        dataset = WikiText2()

    # 1B parameters
    model = Transformer(vocab_size=dataset.vocab_size, nlayers=32, nhid=4096, ninp=1024, nhead=64)

    model = fabric.setup(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer = fabric.setup_optimizers(optimizer)


    for i in range(10):
        input, target = fabric.to_device(dataset[i])
        output = model(input.unsqueeze(0), target.unsqueeze(0))
        loss = F.nll_loss(output, target.view(-1))
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        fabric.print(loss.item())

    fabric.print(torch.cuda.memory_summary())


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

    fabric = L.Fabric(..., strategy=strategy)

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
     - 26’953
     - 11’578
   * - Iteration time (sec)
     - 0.26
     - 0.36

----


*****************************
Speed up model initialization
*****************************

The standard practice in PyTorch is to put all model parameters into CPU memory first and then in a second step move them to the GPU device.
However, the larger the model the longer these two steps take. With the :meth:`~lightning.fabric.fabric.Fabric.init_module` context manager, you can initialize very large models quickly and reduce memory peaks.

Before:

.. code-block:: python

    # Slow: Places the model on CPU first
    model = Transformer(vocab_size=dataset.vocab_size)

After:

.. code-block:: python

    # Fast: Creates the model on the GPU directly
    with fabric.init_module():
        model = Transformer(vocab_size=dataset.vocab_size)

    # Recommended for FSDP:
    with fabric.init_module(empty_init=True):
        model = Transformer(vocab_size=dataset.vocab_size)

For FSDP specifically, we recommend setting ``empty_init=True`` as it will allow you to initialize even larger models.
Empty-init creates fake parameters that don't allocate any memory, their actual initialization gets delayed until ``Fabric.setup()`` where FSDP will shard and recreate the real parameters.
For more use cases of ``empty_init=True`` outside of FSDP, read the guide on :doc:`model initialization <../model_init>`.


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
    fabric = L.Fabric(..., strategy=strategy)


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
     - 26’953
     - 23’181
     - 11’815
     - 11’578
   * - Iteration time (sec)
     - 0.26
     - 0.30
     - 0.31
     - 0.36


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
    fabric = L.Fabric(..., strategy=strategy)


As in our example, it is typical to set the ``activation_checkpointing_policy`` the same as ``auto_wrap_policy``.


Offload parameters to CPU
=========================

The most drastic GPU memory savings can be achieved by offloading parameters to the CPU:

.. code-block:: python

    # Set `cpu_offload=True`
    strategy = FSDPStrategy(..., cpu_offload=True)
    fabric = L.Fabric(..., strategy=strategy)

The drawback is a much slower training speed due to the added communication between CPU and GPU for transferring parameters in every forward pass.
You should use this only if you have enough CPU memory and other scaling methods don’t give you enough memory savings.
In our example, we see a 4x memory saving, but a 10x increase in iteration time:

.. list-table:: Numbers were produced with A100 40GB GPUs, Lightning 2.1 and PyTorch 2.1.
   :widths: 25 25 25 25
   :header-rows: 1

   * -
     - DDP
     - FSDP
     - FSDP + CPU offload
   * - Memory (MB)
     - 26’953
     - 11’578
     - 2’825
   * - Iteration time (sec)
     - 0.26
     - 0.36
     - 3.24


----


*****************
Save a checkpoint
*****************

Since training large models can be very expensive, it is best practice to include checkpointing logic into the training loop to save the progress periodically in case it gets interrupted unexpectedly.
Fabric offers a convenient and efficient method to save large model checkpoints and other state to a checkpoint file.
Simply add the following calls to your training loop:

.. code-block:: python

    # 1. Define model, optimizer, and other training loop state
    state = {"model": model, "optimizer": optimizer, "iter": iteration}

    # DON'T do this (inefficient):
    # state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), ...}

    # 2. Save using Fabric's method
    fabric.save("path/to/checkpoint/file", state)

    # DON'T do this (inefficient):
    # torch.save("path/to/checkpoint/file", state)

To reduce memory peaks and speed up the saving to disk, each process/GPU will save its own file into a folder at the given path by default.
The resulting checkpoint folder will have this structure:

.. code-block:: text

    path/to/checkpoint/file
    ├── .metadata
    ├── __0_0.distcp
    ├── __1_0.distcp
    ...
    └── meta.pt

The “sharded” checkpoint format is the most efficient to save and load in Fabric.
However, if you prefer to have a single consolidated file instead, you can configure this by setting the ``state_dict_type`` flag in the strategy:

.. code-block:: python

    # Default: Save individual files with state from each process
    strategy = FSDPStrategy(state_dict_type="sharded")

    # Save a single, consolidated checkpoint file
    strategy = FSDPStrategy(state_dict_type="full")


**Which checkpoint format should I use?**

- ``state_dict_type="sharded"``: Use for pre-training very large models. It is fast and uses less memory, but it is less portable. An extra step is needed to :doc:`convert the sharded checkpoint into a regular checkpoint file <../../guide/checkpoint/distributed_checkpoint>`.
- ``state_dict_type="full"``: Use when pre-training small to moderately large models (less than 10B parameters), when fine-tuning, and when portability is required.


----


*****************
Load a checkpoint
*****************

You can easily load checkpoints saved by Fabric to resume training:

.. code-block:: python

    # 1. Define model, optimizer, and other training loop state
    state = {"model": model, "optimizer": optimizer, "iter": iteration}

    # 2. Load using Fabric's method
    fabric.load("path/to/checkpoint/file", state)

    # DON'T do this (inefficient):
    # model.load_state_dict(torch.load("path/to/checkpoint/file"))

Fabric will automatically recognize whether the provided path contains a checkpoint saved with ``state_dict_type="full"`` or ``state_dict_type="sharded"``.
Checkpoints saved with ``state_dict_type="full"`` can be loaded by all strategies, but sharded checkpoints can only be loaded by FSDP.
Read :doc:`the checkpoints guide <../../guide/checkpoint/index>` to explore more features.


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
    fabric = L.Fabric(..., strategy=strategy)

You can monitor CUDA malloc retries in the output of ``torch.cuda.memory_summary()`` for example, or through the PyTorch profiler.
