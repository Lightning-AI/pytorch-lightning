Advanced GPU Optimized Training
===============================

When training large models, fitting larger batch sizes, or trying to increase throughput using multi-GPU compute, Lightning provides advanced optimized distributed training plugins to support these cases and offer substantial improvements in memory usage.

Note that some of the extreme memory saving configurations will affect the speed of training. This Speed/Memory trade-off in most cases can be adjusted.

Some of these memory-efficient plugins rely on offloading onto other forms of memory, such as CPU RAM or NVMe. This means you can even see memory benefits on a **single GPU**, using a plugin such as :ref:`deepspeed-zero-stage-3-offload`.

Choosing an Advanced Distributed GPU Plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you would like to stick with PyTorch DDP, see :ref:`ddp-optimizations`.

Unlike PyTorch's DistributedDataParallel (DDP) where the maximum trainable model size and batch size do not change with respect to the number of GPUs, memory-optimized plugins can accommodate bigger models and larger batches as more GPUs are used. This means as you scale up the number of GPUs, you can reach the number of model parameters you'd like to train.

Pre-training vs Fine-tuning
"""""""""""""""""""""""""""

When fine-tuning, we often use a magnitude less data compared to pre-training a model. This is important when choosing a distributed plugin as usually for pre-training, **where we are compute-bound**.
This means we cannot sacrifice throughput as much as if we were fine-tuning, because in fine-tuning the data requirement is smaller.

Overall:

* When **fine-tuning** a model, use advanced memory efficient plugins such as :ref:`deepspeed-zero-stage-3` or :ref:`deepspeed-zero-stage-3-offload`, allowing you to fine-tune larger models if you are limited on compute
* When **pre-training** a model, use simpler optimizations such :ref:`sharded`, :ref:`deepspeed-zero-stage-2`, scaling the number of GPUs to reach larger parameter sizes
* For both fine-tuning and pre-training, use :ref:`deepspeed-activation-checkpointing` or :ref:`fairscale-activation-checkpointing` as the throughput degradation is not significant

For example when using 128 GPUs, you can **pre-train** large 10 to 20 Billion parameter models using :ref:`deepspeed-zero-stage-2` without having to take a performance hit with more advanced optimized multi-gpu plugins.

But for **fine-tuning** a model, you can reach 10 to 20 Billion parameter models using :ref:`deepspeed-zero-stage-3-offload` on a **single GPU**. This does come with a significant throughput hit, which needs to be weighed accordingly.

When Shouldn't I use an Optimized Distributed Plugin?
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Sharding techniques help when model sizes are fairly large; roughly 500M+ parameters is where we've seen benefits. However, in cases where your model is small (ResNet50 of around 80M Parameters) it may be best to stick to ordinary distributed training, unless you are using unusually large batch sizes or inputs.

----------

.. _sharded:

Sharded Training
^^^^^^^^^^^^^^^^
Lightning integration of optimizer sharded training provided by `FairScale <https://github.com/facebookresearch/fairscale>`_.
The technique can be found within `DeepSpeed ZeRO <https://arxiv.org/abs/1910.02054>`_ and
`ZeRO-2 <https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/>`_,
however the implementation is built from the ground up to be pytorch compatible and standalone.
Sharded Training allows you to maintain GPU scaling efficiency, whilst reducing memory overhead drastically. In short, expect near-normal linear scaling (if your network allows), and significantly reduced memory usage when training large models.

Sharded Training still utilizes Data Parallel Training under the hood, except optimizer states and gradients are sharded across GPUs.
This means the memory overhead per GPU is lower, as each GPU only has to maintain a partition of your optimizer state and gradients.

The benefits vary by model and parameter sizes, but we've recorded up to a 63% memory reduction per GPU allowing us to double our model sizes. Because of efficient communication,
these benefits in multi-GPU setups are almost free and throughput scales well with multi-node setups.

It is highly recommended to use Sharded Training in multi-GPU environments where memory is limited, or where training larger models are beneficial (500M+ parameter models).
A technical note: as batch size scales, storing activations for the backwards pass becomes the bottleneck in training. As a result, sharding optimizer state and gradients becomes less impactful.
Use :ref:`fairscale-activation-checkpointing` to see even more benefit at the cost of some throughput.

To use Sharded Training, you need to first install FairScale using the command below.

.. code-block:: bash

    pip install fairscale


.. code-block:: python

    # train using Sharded DDP
    trainer = Trainer(plugins='ddp_sharded')

Sharded Training can work across all DDP variants by adding the additional ``--plugins ddp_sharded`` flag.

Internally we re-initialize your optimizers and shard them across your machines and processes. We handle all communication using PyTorch distributed, so no code changes are required.

.. _fairscale-activation-checkpointing:

FairScale Activation Checkpointing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Activation checkpointing frees activations from memory as soon as they are not needed during the forward pass. They are then re-computed for the backwards pass as needed.

FairScales' checkpointing wrapper also handles batch norm layers correctly unlike the PyTorch implementation, ensuring stats are tracked correctly due to the multiple forward passes.

This saves memory when training larger models however requires wrapping modules you'd like to use activation checkpointing on. See `here <https://fairscale.readthedocs.io/en/latest/api/nn/misc/checkpoint_activations.html>`__ for more information.

.. code-block:: python

    from pytorch_lightning import Trainer
    from fairscale.nn import checkpoint_wrapper


    class MyModel(pl.LightningModule):
        def __init__(self):
            # Wrap layers using checkpoint_wrapper
            self.block = checkpoint_wrapper(nn.Sequential(nn.Linear(32, 32), nn.ReLU()))


.. _deepspeed:

DeepSpeed
^^^^^^^^^

.. note::
    The DeepSpeed plugin is in beta and the API is subject to change. Please create an `issue <https://github.com/PyTorchLightning/pytorch-lightning/issues>`_ if you run into any issues.

`DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ is a deep learning training optimization library, providing the means to train massive billion parameter models at scale.
Using the DeepSpeed plugin, we were able to **train model sizes of 10 Billion parameters and above**, with a lot of useful information in this `benchmark <https://github.com/huggingface/transformers/issues/9996>`_ and the `DeepSpeed docs <https://www.deepspeed.ai/tutorials/megatron/>`__.
DeepSpeed also offers lower level training optimizations, and efficient optimizers such as `1-bit Adam <https://www.deepspeed.ai/tutorials/onebit-adam/>`_. We recommend using DeepSpeed in environments where speed and memory optimizations are important (such as training large billion parameter models).

Below is a summary of all the configurations of DeepSpeed.

* :ref:`deepspeed-zero-stage-2` - **Shard optimizer states and gradients**, remains at parity with DDP with memory improvement

* :ref:`deepspeed-zero-stage-2-offload` - **Offload optimizer states and gradients to CPU**. Increases communication, but significant memory improvement

* :ref:`deepspeed-zero-stage-3` - **Shard optimizer states, gradients, (Optional) activations and parameters**. Increases communication volume, but even more memory improvement

* :ref:`deepspeed-zero-stage-3-offload` - **Offload optimizer states, gradients, (Optional) activations and parameters to CPU**. Increases communication, but even more signficant memory improvement.

* :ref:`deepspeed-activation-checkpointing` - **Free activations after forward pass**. Increases computation, but provides memory improvement for all stages.

To use DeepSpeed, you first need to install DeepSpeed using the commands below.

.. code-block:: bash

    pip install deepspeed

If you run into an issue with the install or later in training, ensure that the CUDA version of the pytorch you've installed matches your locally installed CUDA (you can see which one has been recognized by running ``nvcc --version``).

.. note::

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
    trainer = Trainer(gpus=4, plugins='deepspeed_stage_2', precision=16)
    trainer.fit(model)

.. code-block:: bash

    python train.py --plugins deepspeed_stage_2 --precision 16 --gpus 4


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
    trainer = Trainer(gpus=4, plugins='deepspeed_stage_2_offload', precision=16)
    trainer.fit(model)


This can also be done via the command line using a Pytorch Lightning script:

.. code-block:: bash

    python train.py --plugins deepspeed_stage_2_offload --precision 16 --gpus 4


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
    trainer = Trainer(gpus=4, plugins='deepspeed_stage_2_offload' precision=16)
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

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DeepSpeedPlugin
    from deepspeed.ops.adam import FusedAdam

    class MyModel(pl.LightningModule):
        ...
        def configure_optimizers(self):
            return FusedAdam(self.parameters())

    model = MyModel()
    trainer = Trainer(gpus=4, plugins='deepspeed_stage_3', precision=16)
    trainer.fit(model)

    trainer.test()
    trainer.predict()


Shard Model Instantly to Reduce Initialization Time/Memory
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

When instantiating really large models, it is sometimes necessary to shard the model layers instantly.

This is the case if layers may not fit on one single machines CPU or GPU memory, but would fit once sharded across multiple machines.
We expose a hook that layers initialized within the hook will be sharded instantly on a per layer basis, allowing you to instantly shard models.

This reduces the time taken to initialize very large models, as well as ensure we do not run out of memory when instantiating larger models. For more information you can refer to the DeepSpeed docs for `Constructing Massive Models <https://deepspeed.readthedocs.io/en/latest/zero3.html>`_.

.. code-block:: python

    import torch.nn as nn
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
    trainer = Trainer(gpus=4, plugins='deepspeed_stage_3', precision=16)
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
    trainer = Trainer(gpus=4, plugins='deepspeed_stage_3_offload', precision=16)
    trainer.fit(model)

    # Enable CPU Offloading, and offload parameters to CPU
    model = MyModel()
    trainer = Trainer(
        gpus=4,
        plugins=DeepSpeedPlugin(stage=3, cpu_offload=True, cpu_offload_params=True),
        precision=16
    )
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
        plugins='deepspeed_stage_3_offload',
        precision=16
    )

    # Enable CPU Activation Checkpointing
    trainer = Trainer(
        gpus=4,
        plugins=DeepSpeedPlugin(
            stage=3,
            cpu_offload=True,  # Enable CPU Offloading
            cpu_checkpointing=True  # (Optional) offload activations to CPU
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
* We also support sharded checkpointing. By passing ``save_full_weights=False`` to the ``DeepSpeedPlugin``, we'll save shards of the model which allows you to save extremely large models. However to load the model and run test/validation/predict you must use the Trainer object.

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

.. _ddp-optimizations:

DDP Optimizations
^^^^^^^^^^^^^^^^^


Gradients as Bucket View
""""""""""""""""""""""""

Enabling ``gradient_as_bucket_view=True`` in the ``DDPPlugin`` will make gradients views point to different offsets of the ``allreduce`` communication buckets. See `DistributedDataParallel <https://pytorch.org/docs/master/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel>`__ for more information.

This can reduce peak memory usage and throughput as saved memory will be equal to the total gradient memory + removes the need to copy gradients to the ``allreduce`` communication buckets.

.. note::

    When ``gradient_as_bucket_view=True`` you cannot call ``detach_()`` on gradients. If hitting such errors, please fix it by referring to the :meth:`~torch.optim.Optimizer.zero_grad` function in ``torch/optim/optimizer.py`` as a solution (`source <https://pytorch.org/docs/master/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel>`__).

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DDPPlugin

    model = MyModel()
    trainer = Trainer(gpus=4, plugins=DDPPlugin(gradient_as_bucket_view=True))
    trainer.fit(model)

DDP Communication Hooks
"""""""""""""""""""""""

DDP Communication hooks is an interface to control how gradients are communicated across workers, overriding the standard allreduce in DistributedDataParallel. This allows you to enable performance improving communication hooks when using multiple nodes.

.. note::
    DDP communication hooks needs pytorch version at least 1.8.0

Enable `FP16 Compress Hook for multi-node throughput improvement <https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook>`__:

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DDPPlugin
    from torch.distributed.algorithms.ddp_comm_hooks import (
            default_hooks as default,
            powerSGD_hook as powerSGD,
    )

    model = MyModel()
    trainer = Trainer(gpus=4, plugins=DDPPlugin(ddp_comm_hook=default.fp16_compress_hook))
    trainer.fit(model)

Enable `PowerSGD for multi-node throughput improvement <https://pytorch.org/docs/stable/ddp_comm_hooks.html#powersgd-communication-hook>`__:

.. note::

    PowerSGD typically requires extra memory of the same size as the model’s gradients to enable error feedback, which can compensate for biased compressed communication and improve accuracy (`source <https://pytorch.org/docs/stable/ddp_comm_hooks.html#powersgd-hooks>`__).

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DDPPlugin
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD

    model = MyModel()
    trainer = Trainer(
        gpus=4,
        plugins=DDPPlugin(
            ddp_comm_state=powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=5000,
            ),
            ddp_comm_hook=powerSGD.powerSGD_hook,
        )
    )
    trainer.fit(model)


Combine hooks for accumulated benefit:

.. note::
    DDP communication wrappers needs pytorch version at least 1.9.0

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DDPPlugin
    from torch.distributed.algorithms.ddp_comm_hooks import (
            default_hooks as default,
            powerSGD_hook as powerSGD,
    )

    model = MyModel()
    trainer = Trainer(
        gpus=4,
        plugins=DDPPlugin(
            ddp_comm_state=powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=5000,
            ),
            ddp_comm_hook=powerSGD.powerSGD_hook,
            ddp_comm_wrapper=default.fp16_compress_wrapper,
        )
    )
    trainer.fit(model)
