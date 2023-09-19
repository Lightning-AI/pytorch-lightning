:orphan:

.. _ddp-optimizations:

#################
DDP Optimizations
#################

Tune settings specific to DDP training for increased speed and memory efficiency.


----


***********************
Gradient as Bucket View
***********************

Enabling ``gradient_as_bucket_view=True`` in the ``DDPStrategy`` will make gradients views point to different offsets of the ``allreduce`` communication buckets.
See :class:`~torch.nn.parallel.DistributedDataParallel` for more information.
This can reduce peak memory usage and throughput as saved memory will be equal to the total gradient memory + removes the need to copy gradients to the ``allreduce`` communication buckets.

.. code-block:: python

    import lightning as L
    from lightning.pytorch.strategies import DDPStrategy

    model = MyModel()
    trainer = L.Trainer(devices=4, strategy=DDPStrategy(gradient_as_bucket_view=True))
    trainer.fit(model)

.. note::
    When ``gradient_as_bucket_view=True`` you cannot call ``detach_()`` on gradients.


----


****************
DDP Static Graph
****************

`DDP static graph <https://pytorch.org/blog/pytorch-1.11-released/#stable-ddp-static-graph>`__ assumes that your model employs the same set of used/unused parameters in every iteration, so that it can deterministically know the flow of training and apply special optimizations during runtime.

.. code-block:: python

    import lightning as L
    from lightning.pytorch.strategies import DDPStrategy

    trainer = L.Trainer(devices=4, strategy=DDPStrategy(static_graph=True))


----


********************************************
On a Multi-Node Cluster, Set NCCL Parameters
********************************************

`NCCL <https://developer.nvidia.com/nccl>`__ is the NVIDIA Collective Communications Library that is used by PyTorch to handle communication across nodes and GPUs.
There are reported benefits in terms of speedups when adjusting NCCL parameters as seen in this `issue <https://github.com/Lightning-AI/lightning/issues/7179>`__.
In the issue, we see a 30% speed improvement when training the Transformer XLM-RoBERTa and a 15% improvement in training with Detectron2.
NCCL parameters can be adjusted via environment variables.

.. note::

    AWS and GCP already set default values for these on their clusters.
    This is typically useful for custom cluster setups.

* `NCCL_NSOCKS_PERTHREAD <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nsocks-perthread>`__
* `NCCL_SOCKET_NTHREADS <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-nthreads>`__
* `NCCL_MIN_NCHANNELS <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-min-nchannels>`__

.. code-block:: bash

    export NCCL_NSOCKS_PERTHREAD=4
    export NCCL_SOCKET_NTHREADS=2


----


***********************
DDP Communication Hooks
***********************

DDP Communication hooks is an interface to control how gradients are communicated across workers, overriding the standard allreduce in :class:`~torch.nn.parallel.DistributedDataParallel`.
This allows you to enable performance improving communication hooks when using multiple nodes.
Enable `FP16 Compress Hook for multi-node throughput improvement <https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook>`__:

.. code-block:: python

    import lightning as L
    from lightning.pytorch.strategies import DDPStrategy
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

    model = MyModel()
    trainer = L.Trainer(accelerator="gpu", devices=4, strategy=DDPStrategy(ddp_comm_hook=default.fp16_compress_hook))
    trainer.fit(model)

Enable `PowerSGD for multi-node throughput improvement <https://pytorch.org/docs/stable/ddp_comm_hooks.html#powersgd-communication-hook>`__:

.. note::

    PowerSGD typically requires extra memory of the same size as the model’s gradients to enable error feedback, which can compensate for biased compressed communication and improve accuracy (`source <https://pytorch.org/docs/stable/ddp_comm_hooks.html#powersgd-hooks>`__).

.. code-block:: python

    import lightning as L
    from lightning.pytorch.strategies import DDPStrategy
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD

    model = MyModel()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DDPStrategy(
            ddp_comm_state=powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=5000,
            ),
            ddp_comm_hook=powerSGD.powerSGD_hook,
        ),
    )
    trainer.fit(model)


Combine hooks for accumulated benefit:

.. code-block:: python

    import lightning as L
    from lightning.pytorch.strategies import DDPStrategy
    from torch.distributed.algorithms.ddp_comm_hooks import (
        default_hooks as default,
        powerSGD_hook as powerSGD,
    )

    model = MyModel()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DDPStrategy(
            ddp_comm_state=powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=5000,
            ),
            ddp_comm_hook=powerSGD.powerSGD_hook,
            ddp_comm_wrapper=default.fp16_compress_wrapper,
        ),
    )
    trainer.fit(model)


When using Post-localSGD, you must also pass ``model_averaging_period`` to allow for model parameter averaging:

.. code-block:: python

    import lightning as L
    from lightning.pytorch.strategies import DDPStrategy
    from torch.distributed.algorithms.ddp_comm_hooks import post_localSGD_hook as post_localSGD

    model = MyModel()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DDPStrategy(
            ddp_comm_state=post_localSGD.PostLocalSGDState(
                process_group=None,
                subgroup=None,
                start_localSGD_iter=8,
            ),
            ddp_comm_hook=post_localSGD.post_localSGD_hook,
            model_averaging_period=4,
        ),
    )
    trainer.fit(model)
