.. _performance:

Fast performance tips
=====================
Lightning builds in all the micro-optimizations we can find to increase your performance.
But we can only automate so much.

Here are some additional things you can do to increase your performance.

----------

Dataloaders
-----------
When building your DataLoader set ``num_workers > 0`` and ``pin_memory=True`` (only for GPUs).

.. code-block:: python

    Dataloader(dataset, num_workers=8, pin_memory=True)

num_workers
^^^^^^^^^^^
The question of how many ``num_workers`` is tricky. Here's a summary of
some references, [`1 <https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813>`_], and our suggestions.

1. ``num_workers=0`` means ONLY the main process will load batches (that can be a bottleneck).
2. ``num_workers=1`` means ONLY one worker (just not the main process) will load data but it will still be slow.
3. The ``num_workers`` depends on the batch size and your machine.
4. A general place to start is to set ``num_workers`` equal to the number of CPUs on that machine.

.. warning:: Increasing ``num_workers`` will ALSO increase your CPU memory consumption.

The best thing to do is to increase the ``num_workers`` slowly and stop once you see no more improvement in your training speed.

Spawn
^^^^^
When using ``accelerator=ddp_spawn`` (the ddp default) or TPU training, the way multiple GPUs/TPU cores are used is by calling ``.spawn()`` under the hood.
The problem is that PyTorch has issues with ``num_workers > 0`` when using ``.spawn()``. For this reason we recommend you
use ``accelerator=ddp`` so you can increase the ``num_workers``, however your script has to be callable like so:

.. code-block:: bash

    python my_program.py --gpus X

----------

.item(), .numpy(), .cpu()
-------------------------
Don't call ``.item()`` anywhere in your code. Use ``.detach()`` instead to remove the connected graph calls. Lightning
takes a great deal of care to be optimized for this.

----------

empty_cache()
-------------
Don't call this unnecessarily! Every time you call this ALL your GPUs have to wait to sync.

----------

Construct tensors directly on the device
----------------------------------------
LightningModules know what device they are on! Construct tensors on the device directly to avoid CPU->Device transfer.

.. code-block:: python

    # bad
    t = torch.rand(2, 2).cuda()

    # good (self is LightningModule)
    t = torch.rand(2, 2, device=self.device)


For tensors that need to be model attributes, it is best practice to register them as buffers in the modules's
``__init__`` method:

.. code-block:: python

    # bad
    self.t = torch.rand(2, 2, device=self.device)

    # good
    self.register_buffer("t", torch.rand(2, 2))

----------

Use DDP not DP
--------------
DP performs three GPU transfers for EVERY batch:

1. Copy model to device.
2. Copy data to device.
3. Copy outputs of each device back to master.

|

Whereas DDP only performs 1 transfer to sync gradients. Because of this, DDP is MUCH faster than DP.

----------

16-bit precision
----------------
Use 16-bit to decrease the memory consumption (and thus increase your batch size). On certain GPUs (V100s, 2080tis), 16-bit calculations are also faster.
However, know that 16-bit and multi-processing (any DDP) can have issues. Here are some common problems.

1. `CUDA error: an illegal memory access was encountered <https://github.com/pytorch/pytorch/issues/21819>`_.
    The solution is likely setting a specific CUDA, CUDNN, PyTorch version combination.
2. ``CUDA error: device-side assert triggered``. This is a general catch-all error. To see the actual error run your script like so:

.. code-block:: bash

    # won't see what the error is
    python main.py

    # will see what the error is
    CUDA_LAUNCH_BLOCKING=1 python main.py

.. tip:: We also recommend using 16-bit native found in PyTorch 1.6. Just install this version and Lightning will automatically use it.

----------

Use Sharded DDP for GPU memory and scaling optimization
-------------------------------------------------------

Sharded DDP is a lightning integration of `DeepSpeed ZeRO <https://arxiv.org/abs/1910.02054>`_ and
`ZeRO-2 <https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/>`_
provided by `Fairscale <https://github.com/facebookresearch/fairscale>`_.

When training on multiple GPUs sharded DDP can assist to increase memory efficiency substantially, and in some cases performance on multi-node is better than traditional DDP.
This is due to efficient communication and parallelization under the hood.

To use Optimizer Sharded Training, refer to :ref:`model-parallelism`.

Sharded DDP can work across all DDP variants by adding the additional ``--plugins ddp_sharded`` flag.

Refer to the :ref:`distributed computing guide for more details <multi_gpu>`.


Sequential Model Parallelism with Checkpointing
---------------------------------------------------------------------
PyTorch Lightning integration for Sequential Model Parallelism using `FairScale <https://github.com/facebookresearch/fairscale>`_.
Sequential Model Parallelism splits a sequential module onto multiple GPUs, reducing peak GPU memory requirements substantially.

For more information, refer to :ref:`sequential-parallelism`.
