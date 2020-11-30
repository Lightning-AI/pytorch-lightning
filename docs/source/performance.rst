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


Pipeline Parallelism with Checkpointing to reduce peak memory
-------------------------------------------------------------

Pipe Pipeline is a lightning integration of Pipeline Parallelism provided by Fairscale.

Pipe combines pipeline parallelism with checkpointing to reduce peak memory required to train while minimizing device under-utilization.

Before running, install Fairscale using the command below or install all extras using pip install pytorch-lightning["extra"].

or

```
pip install https://github.com/facebookresearch/fairscale/archive/master.zip
```

We except the nn.Sequential model to be set as `.layers` attribute to your LightningModule.


.. code-block:: bash

    from pytorch_lightning.plugins.pipe_plugin import PipePlugin

    class MyModel(LightningModule):

        def __init__(...):

            self.layers = nn.Sequential(torch.nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2)) # 3 layers

        ....

    model = MyModel()

    # train by balancing your 2 first layers on gpu 0 and last layer gpu 1
    trainer = Trainer(accelerator='ddp', plugins=PipePlugin(balance=[2, 1]))

    trainer.fit(model)


With auto-balancing.

By `example_input_array`, we can infer automatically the right balance for your model.

.. code-block:: bash

    from pytorch_lightning.plugins.pipe_plugin import PipePlugin

    class MyModel(LightningModule):

        def __init__(...):

            self.layers = nn.Sequential(torch.nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2)) # 3 layers

            # used to make an inference and find best balancing for your model
            self._example_input_array = torch.randn((1, 32))

        ....

    model = MyModel()

    # train by balancing your 2 first layers on gpu 0 and last layer gpu 1
    trainer = Trainer(accelerator='ddp', plugins='pipe')

    trainer.fit(model)
