Memory Optimized Multi-GPU Training
===================================

When you want to train larger parameter models or fit larger batch sizes on your multi-gpu compute, Lightning provides advanced optimized distributed training to support these cases.

For example if you'd like to train a large billion parameter transformer model, or to scale your batch size when training a semi-supervised learning model, using a Lightning optimized distributed training plugin will offer substantial improvements
in memory usage. Note that some of the extreme memory saving configurations will affect the speed of training. This Speed/Memory trade-off in most cases can be adjusted.

Choosing an Optimized Distributed Plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These optimized Multi-GPU plugins shard the model states across your GPUs; just in different ways. This means as you scale up the number of GPUs,
you may be able to reach the number of model parameters you'd like to train using plugins that have less of a speed degradation.

For example when using 128 GPUs, you can scale to large 10 to 20 Billion parameter models using just DeepSpeed ZeRO Stage 2.

- I want to reach the largest **batch size**, with *minimal* speed degradation - Use :ref:`sharded` or :ref:`deepspeed-zero-stage-2`
- I want to reach the largest **model size**, with *minimal* speed degradation - Use :ref:`deepspeed-zero-stage-2`
- I want to reach the largest **batch size**, and I don't mind a *small* speed hit - Use :ref:`deepspeed-zero-stage-3`
- I want to reach the largest **model size**, and I don't mind a *small* speed hit - Use :ref:`deepspeed-zero-stage-3`
- I want to reach the largest **batch size**, and I don't mind a speed hit - Use :ref:`deepspeed-zero-stage-3-offload` and :ref:`deepspeed-activation-checkpointing`
- I want to reach the largest **model size**, and I don't mind a speed hit - Use :ref:`deepspeed-zero-stage-3-offload` and :ref:`deepspeed-activation-checkpointing`

.. _sharded:

Sharded Training
^^^^^^^^^^^^^^^^
Lightning integration of optimizer sharded training provided by `FairScale <https://github.com/facebookresearch/fairscale>`_.
The technique can be found within `DeepSpeed ZeRO <https://arxiv.org/abs/1910.02054>`_ and
`ZeRO-2 <https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/>`_,
however the implementation is built from the ground up to be pytorch compatible and standalone.
Sharded Training allows you to maintain GPU scaling efficiency, whilst reducing memory overhead drastically. In short, expect normal linear scaling, and significantly reduced memory usage when training large models.

Sharded Training still utilizes Data Parallel Training under the hood, except optimizer states and gradients are sharded across GPUs.
This means the memory overhead per GPU is lower, as each GPU only has to maintain a partition of your optimizer state and gradients.

The benefits vary by model and parameter sizes, but we've recorded up to a 63% memory reduction per GPU allowing us to double our model sizes. Because of extremely efficient communication,
these benefits in multi-GPU setups are almost free and throughput scales well with multi-node setups.

Below we use the `NeMo Transformer Lightning Language Modeling example <https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/language_modeling>`_ to benchmark the maximum batch size and model size that can be fit on 8 A100 GPUs for DDP vs Sharded Training.
Note that the benefits can still be obtained using 2 or more GPUs, and for even larger batch sizes you can scale to multiple nodes.

**Increase Your Batch Size**

Use Sharded Training to scale your batch size further using the same compute. This will reduce your overall epoch time.

+----------------------+-----------------------+----------------+---------------------+
| Distributed Training | Model Size (Millions) | Max Batch Size | Percentage Gain (%) |
+======================+=======================+================+=====================+
| Native DDP           | 930                   | 32             | -                   |
+----------------------+-----------------------+----------------+---------------------+
| Sharded DDP          | 930                   | **52**         | **48%**             |
+----------------------+-----------------------+----------------+---------------------+

**Increase Your Model Size**

Use Sharded Training to scale your model size further using the same compute.

+----------------------+------------+---------------------------+---------------------+
| Distributed Training | Batch Size | Max Model Size (Millions) | Percentage Gain (%) |
+======================+============+===========================+=====================+
| Native DDP           | 32         | 930                       | -                   |
+----------------------+------------+---------------------------+---------------------+
| Sharded DDP          | 32         | **1404**                  | **41%**             |
+----------------------+------------+---------------------------+---------------------+
| Native DDP           | 8          | 1572                      | -                   |
+----------------------+------------+---------------------------+---------------------+
| Sharded DDP          | 8          | **2872**                  | **59%**             |
+----------------------+------------+---------------------------+---------------------+

It is highly recommended to use Sharded Training in multi-GPU environments where memory is limited, or where training larger models are beneficial (500M+ parameter models).
A technical note: as batch size scales, storing activations for the backwards pass becomes the bottleneck in training. As a result, sharding optimizer state and gradients becomes less impactful.
Work within the future will bring optional sharding to activations and model parameters to reduce memory further, but come with a speed cost.

To use Sharded Training, you need to first install FairScale using the command below.

.. code-block:: bash

    pip install fairscale


.. code-block:: python

    # train using Sharded DDP
    trainer = Trainer(accelerator='ddp', plugins='ddp_sharded')

Sharded Training can work across all DDP variants by adding the additional ``--plugins ddp_sharded`` flag.

Internally we re-initialize your optimizers and shard them across your machines and processes. We handle all communication using PyTorch distributed, so no code changes are required.

----------

.. _deep_speed:

DeepSpeed
^^^^^^^^^

.. note::
    The DeepSpeed plugin is in beta and the API is subject to change. Please create an `issue <https://github.com/PyTorchLightning/pytorch-lightning/issues>`_ if you run into any issues.

`DeepSpeed <https://github.com/microsoft/DeepSpeed>`_ is a deep learning training optimization library, providing the means to train massive billion parameter models at scale.
Using the DeepSpeed plugin, we were able to **train model sizes of 10 Billion parameters and above**, with a lot of useful information in this `benchmark <https://github.com/huggingface/transformers/issues/9996>`_ and the DeepSpeed `docs <https://www.deepspeed.ai/tutorials/megatron/>`_.
DeepSpeed also offers lower level training optimizations, and efficient optimizers such as `1-bit Adam <https://www.deepspeed.ai/tutorials/onebit-adam/>`_. We recommend using DeepSpeed in environments where speed and memory optimizations are important (such as training large billion parameter models).

To use DeepSpeed, you first need to install DeepSpeed using the commands below.

.. code-block:: bash

    pip install deepspeed

If you run into an issue with the install or later in training, ensure that the CUDA version of the pytorch you've installed matches your locally installed CUDA (you can see which one has been recognized by running ``nvcc --version``).

.. note::
    Currently ``resume_from_checkpoint`` and manual optimization are not supported.

    DeepSpeed currently only supports single optimizer, single scheduler within the training loop.

.. _deepspeed-zero-stage-2:

DeepSpeed ZeRO Stage 2
""""""""""""""""""""""

By default, we enable `DeepSpeed ZeRO Stage 2 <https://www.deepspeed.ai/tutorials/zero/#zero-overview>`_, which partitions your optimizer states (Stage 1) and your gradients (Stage 2) across your GPUs to reduce memory. In most cases, this is more efficient or at parity with DDP, primarily due to the optimized custom communications written by the DeepSpeed team.
As a result, benefits can also be seen on a single GPU. Do note that the default bucket sizes allocate around ``3.6GB`` of VRAM to use during distributed communications, which can be tweaked when instantiating the plugin described in a few sections below.

.. note::
    To use ZeRO, you must use ``precision=16``.

.. code-block:: python

    from pytorch_lightning import Trainer

    model = MyModel()
    trainer = Trainer(gpus=4, plugins='deepspeed', precision=16)
    trainer.fit(model)

.. _deepspeed-zero-stage-2-offload:

DeepSpeed ZeRO Stage 2 Offload
""""""""""""""""""""""""""""""

Below we show an example of running `ZeRO-Offload <https://www.deepspeed.ai/tutorials/zero-offload/>`_. ZeRO-Offload leverages the host CPU to offload optimizer memory/computation, reducing the overall memory consumption.

.. note::
    To use ZeRO-Offload, you must use ``precision=16``.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DeepSpeedPlugin

    model = MyModel()
    trainer = Trainer(gpus=4, plugins=DeepSpeedPlugin(cpu_offload=True), precision=16)
    trainer.fit(model)


This can also be done via the command line using a Pytorch Lightning script:

.. code-block:: bash

    python train.py --plugins deepspeed --precision 16 --gpus 4


You can also modify the ZeRO-Offload parameters via the plugin as below.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DeepSpeedPlugin

    model = MyModel()
    trainer = Trainer(gpus=4, plugins=DeepSpeedPlugin(cpu_offload=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8), precision=16)
    trainer.fit(model)


.. note::
    We suggest tuning the ``allgather_bucket_size`` parameter and ``reduce_bucket_size`` parameter to find optimum parameters based on your model size.
    These control how large a buffer we limit the model to using when reducing gradients/gathering updated parameters. Smaller values will result in less memory, but tradeoff with speed.

    DeepSpeed allocates a reduce buffer size `multiplied by 4.5x <https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage2.py#L1594-L1607>`_ so take that into consideration when tweaking the parameters.

    The plugin sets a reasonable default of ``2e8``, which should work for most low VRAM GPUs (less than ``7GB``), allocating roughly ``3.6GB`` of VRAM as buffer. Higher VRAM GPUs should aim for values around ``5e8``.

For even more speed benefit, DeepSpeed offers an optimized CPU version of ADAM called `DeepSpeedCPUAdam <https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu>`_ to run the offloaded computation, which is faster than the standard PyTorch implementation.

.. code-block:: python

    import pytorch_lightning
    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DeepSpeedPlugin
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    class MyModel(pl.LightningModule):
        ...
        def configure_optimizers(self):
            # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
            return DeepSpeedCPUAdam(self.parameters())

    model = MyModel()
    trainer = Trainer(gpus=4, plugins=DeepSpeedPlugin(cpu_offload=True), precision=16)
    trainer.fit(model)

.. _deepspeed-zero-stage-3:

DeepSpeed ZeRO Stage 3
""""""""""""""""""""""

DeepSpeed ZeRO Stage 3 shards the optimizer states, gradients and the model parameters (also optionally activations). Sharding model parameters and activations comes with an increase in distributed communication, however allows you to scale your models massively from one GPU to multiple GPUs.
**The DeepSpeed team report the ability to fine-tune models with over 40B parameters on a single GPU and over 2 Trillion parameters on 512 GPUs.** For more information we suggest checking the `DeepSpeed ZeRO-3 Offload documentation <https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html>`__.

We've ran benchmarks for all these features and given a simple example of how all these features work in Lightning, which you can see at `minGPT <https://github.com/SeanNaren/minGPT/tree/stage3>`_.

Currently this functionality is only available on master and will be included in our next 1.3 Release Candidate and 1.3 release.

.. code-block:: python

    pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/refs/heads/master.zip


To reach the highest memory efficiency or model size, you must:

1. Use the DeepSpeed Plugin with the stage 3 parameter
2. Use CPU Offloading to offload weights to CPU, plus have a reasonable amount of CPU RAM to offload onto
3. Use DeepSpeed Activation Checkpointing to shard activations

Below we describe how to enable all of these to see benefit. **With all these improvements we reached 45 Billion parameters training a GPT model on 8 GPUs with ~1TB of CPU RAM available**.

Also please have a look at our :ref:`deepspeed-zero-stage-3-tips` which contains a lot of helpful information when configuring your own models.

.. note::
    Currently we only support non-elastic checkpointing. This means saving the model across GPUs will save shards of the model on all processes, which will then require the same amount of GPUS to load.
    This additionally means for inference you must use the ``Trainer.test`` or ``Trainer.predict`` functionality as described below, to ensure we set up the distributed environment correctly.

    This limitation is actively being worked on and will be resolved in the near future.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DeepSpeedPlugin
    from deepspeed.ops.adam import FusedAdam

    class MyModel(pl.LightningModule):
        ...
        def configure_optimizers(self):
            return FusedAdam(self.parameters())

    model = MyModel()
    trainer = Trainer(gpus=4, plugins=DeepSpeedPlugin(stage=3), precision=16)
    trainer.fit(model)

    trainer.test()
    trainer.predict()


Shard Model Instantly to Reduce Initialization Time/Memory
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

When instantiating really large models, it is sometimes necessary to shard the model layers instantly.

This is the case if layers may not fit on one single machines CPU or GPU memory, but would fit once sharded across multiple machines.
We expose a hook that layers initialized within the hook will be sharded instantly on a per layer basis, allowing you to instantly shard models.

This reduces the time taken to initialize very large models, as well as ensure we do not run out of memory when instantiating larger models. For more information you can refer to the DeepSpeed docs for `Constructing Massive Models <https://deepspeed.readthedocs.io/en/latest/zero3.html>`_.

.. note::
    When using the ``configure_sharded_model`` hook to shard models, note that ``LightningModule.load_from_checkpoint`` may not work for loading saved checkpoints. If you've trained on one GPU, you can manually instantiate the model and call the hook,
    however when using multiple GPUs, this will not work as ``LightningModule.load_from_checkpoint`` doesn't support sharded checkpoints.

    We recommend using ``Trainer.test`` or ``Trainer.predict`` for inference.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DeepSpeedPlugin
    from deepspeed.ops.adam import FusedAdam

    class MyModel(pl.LightningModule):
        ...
        def configure_sharded_model(self):
            # Created within sharded model context, modules are instantly sharded across processes
            # as soon as they are made.
            self.block = nn.Sequential(nn.Linear(32, 32), nn.ReLU())

        def configure_optimizers(self):
            return FusedAdam(self.parameters())

    model = MyModel()
    trainer = Trainer(gpus=4, plugins=DeepSpeedPlugin(stage=3), precision=16)
    trainer.fit(model)

    trainer.test()
    trainer.predict()


.. _deepspeed-zero-stage-3-offload:

DeepSpeed ZeRO Stage 3 Offload
""""""""""""""""""""""""""""""

DeepSpeed ZeRO Stage 3 Offloads optimizer state, gradients to the host CPU to reduce memory usage as ZeRO Stage 2 does, however additionally allows you to offload the parameters as well for even more memory saving.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DeepSpeedPlugin

    # Enable CPU Offloading
    model = MyModel()
    trainer = Trainer(gpus=4, plugins=DeepSpeedPlugin(stage=3, cpu_offload=True), precision=16)
    trainer.fit(model)

    # Enable CPU Offloading, and offload parameters as well to CPU when possible
    model = MyModel()
    trainer = Trainer(gpus=4, plugins=DeepSpeedPlugin(stage=3, cpu_offload=True, cpu_offload_params=True), precision=16)
    trainer.fit(model)


.. _deepspeed-activation-checkpointing:

DeepSpeed Activation Checkpointing
""""""""""""""""""""""""""""""""""

Activation checkpointing frees activations from memory as soon as they are not needed during the forward pass.
They are then re-computed for the backwards pass as needed.

This saves memory when training larger models however requires using a checkpoint function to run the module as shown below.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DeepSpeedPlugin
    import deepspeed


    class MyModel(pl.LightningModule):
        ...

        def configure_sharded_model(self):
            self.block = nn.Sequential(nn.Linear(32, 32), nn.ReLU())

        def forward(self, x):
            # Use the DeepSpeed checkpointing function instead of calling the module directly
            output = deepspeed.checkpointing.checkpoint(self.block, x)
            return output


    model = MyModel()
    trainer = Trainer(
        gpus=4,
        plugins=DeepSpeedPlugin(
            stage=3,
            cpu_offload=True,  # Enable CPU Offloading
            partition_activations=True,  # Optionally move activations to CPU if you have enough memory
            cpu_checkpointing=True  # Optionally Partition activations across machines
        ),
        precision=16
    )
    trainer.fit(model)


.. _deepspeed-zero-stage-3-tips:

DeepSpeed ZeRO Stage 3 Tips
"""""""""""""""""""""""""""

Here is some helpful information when setting up DeepSpeed ZeRO Stage 3 with Lightning.

* If you're using Adam or AdamW, ensure to use FusedAdam or DeepSpeedCPUAdam (for CPU Offloading) rather than the default torch optimizers as they come with large speed benefits
* Treat your GPU/CPU memory as one large pool. In some cases, you may not want to offload certain things (like activations) to provide even more space to offload model parameters
* When offloading to the CPU, make sure to bump up the batch size as GPU memory will be freed


Custom DeepSpeed Config
"""""""""""""""""""""""

In some cases you may want to define your own DeepSpeed Config, to access all parameters defined. We've exposed most of the important parameters, however, there may be debugging parameters to enable. Also, DeepSpeed allows the use of custom DeepSpeed optimizers and schedulers defined within a config file that is supported.

.. note::
    All plugin default parameters will be ignored when a config object is passed.
    All compatible arguments can be seen in the `DeepSpeed docs <https://www.deepspeed.ai/docs/config-json/>`_.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DeepSpeedPlugin

    deepspeed_config = {
        "zero_allow_untested_optimizer": True,
        "optimizer": {
            "type": "OneBitAdam",
            "params": {
                "lr": 3e-5,
                "betas": [0.998, 0.999],
                "eps": 1e-5,
                "weight_decay": 1e-9,
                "cuda_aware": True,
            },
        },
        'scheduler': {
            "type": "WarmupLR",
            "params": {
                "last_batch_iteration": -1,
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 100,
            }
        },
        "zero_optimization": {
            "stage": 2, # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
            "cpu_offload": True, # Enable Offloading optimizer state/calculation to the host CPU
            "contiguous_gradients": True, # Reduce gradient fragmentation.
            "overlap_comm": True, # Overlap reduce/backward operation of gradients for speed.
            "allgather_bucket_size": 2e8, # Number of elements to all gather at once.
            "reduce_bucket_size": 2e8, # Number of elements we reduce/allreduce at once.
        }
    }

    model = MyModel()
    trainer = Trainer(gpus=4, plugins=DeepSpeedPlugin(deepspeed_config), precision=16)
    trainer.fit(model)


We support taking the config as a json formatted file:

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DeepSpeedPlugin

    model = MyModel()
    trainer = Trainer(gpus=4, plugins=DeepSpeedPlugin("/path/to/deepspeed_config.json"), precision=16)
    trainer.fit(model)


You can use also use an environment variable via your PyTorch Lightning script:

.. code-block:: bash

    PL_DEEPSPEED_CONFIG_PATH=/path/to/deepspeed_config.json python train.py --plugins deepspeed


----------

.. _sequential-parallelism:

Sequential Model Parallelism with Checkpointing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch Lightning integration for Sequential Model Parallelism using `FairScale <https://github.com/facebookresearch/fairscale>`_.
Sequential Model Parallelism splits a sequential module onto multiple GPUs, reducing peak GPU memory requirements substantially.
We also provide auto-balancing techniques through FairScale, to find optimal balances for the model across GPUs.
In addition, we use Gradient Checkpointing to reduce GPU memory requirements further, and micro-batches to minimizing device under-utilization automatically.

Reference: https://arxiv.org/abs/1811.06965

.. note:: RPCSequentialPlugin is currently supported only for Pytorch 1.6.

To get started, install FairScale using the command below. We install a specific branch which contains PyTorch related fixes for Sequential Parallelism.

.. code-block:: bash

     pip install https://github.com/PyTorchLightning/fairscale/archive/pl_1.2.0.zip

To use Sequential Model Parallelism, you must define a  :class:`nn.Sequential <torch.nn.Sequential>` module that defines the layers you wish to parallelize across GPUs.
This should be kept within the ``sequential_module`` variable within your ``LightningModule`` like below.

.. code-block:: python

    from pytorch_lightning.plugins.training_type.rpc_sequential import RPCSequentialPlugin
    from pytorch_lightning import LightningModule

    class MyModel(LightningModule):
        def __init__(self):
            ...
            self.sequential_module = nn.Sequential(my_layers)

    # Split my module across 4 gpus, one layer each
    model = MyModel()
    plugin = RPCSequentialPlugin(balance=[1, 1, 1, 1])
    trainer = Trainer(accelerator='ddp', gpus=4, plugins=[plugin])
    trainer.fit(model)


We provide a minimal example of Sequential Model Parallelism using a convolutional model training on cifar10, split onto GPUs `here <https://github.com/PyTorchLightning/pytorch-lightning/tree/master/pl_examples/basic_examples/conv_sequential_example.py>`_.
To run the example, you need to install `Bolts <https://github.com/PyTorchLightning/pytorch-lightning-bolts>`_. Install with ``pip install pytorch-lightning-bolts``.

When running the Sequential Model Parallelism example on 2 GPUS we achieve these memory savings.

.. list-table:: GPU Memory Utilization
   :widths: 25 25 50
   :header-rows: 1

   * - GPUS
     - Without Balancing
     - With Balancing
   * - Gpu 0
     - 4436 MB
     - 1554 MB
   * - Gpu 1
     - ~0
     - 994 MB

To run the example with Sequential Model Parallelism:

.. code-block:: bash

    python pl_examples/basic_examples/conv_sequential_example.py --batch_size 1024 --gpus 2 --accelerator ddp --use_ddp_sequential

To run the same example without Sequential Model Parallelism:

.. code-block:: bash

    python pl_examples/basic_examples/conv_sequential_example.py --batch_size 1024 --gpus 1
