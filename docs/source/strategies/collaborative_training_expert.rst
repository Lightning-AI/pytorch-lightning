:orphan:

.. _collaborative_training_expert:

Training on unreliable mixed GPUs across the internet (Expert)
==============================================================

Using Compression to Optimize communications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below are some ways to reduce communication when training collaboratively. As the model sizes get larger, bottlenecks in communication become more apparent.

Compress Gradients & State
""""""""""""""""""""""""""

Hivemind allows you to compress gradients and state before sending to other machines. This helps reduce the communication overhead substantially when training across the internet.

Below we enable Float16 compression, which compresses gradients and state to Float16 before sending to other machines.

.. code-block:: python

    from hivemind import Float16Compression
    import pytorch_lightning as pl
    from pytorch_lightning.strategies import CollaborativeStrategy

    trainer = pl.Trainer(
        strategy=CollaborativeStrategy(
            target_batch_size=target_batch_size,
            grad_compression=Float16Compression(),
            state_averaging_compression=Float16Compression(),
        ),
        accelerator="gpu",
        devices=1,
    )

A slightly more advanced scheme is dynamic compression based on the size of values. Below we enable 8-bit quantization for large numbers, and Float16 compression for small values.

Size Adaptive Compression has been used in a variety of Hivemind applications and has shown success.

.. code-block:: python

    from hivemind import Float16Compression, Uniform8BitQuantization
    import pytorch_lightning as pl
    from pytorch_lightning.strategies import CollaborativeStrategy

    # compresses values above threshold with 8bit Quantization, lower with Float16
    compression = SizeAdaptiveCompression(
        threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization()
    )
    trainer = pl.Trainer(
        strategy=CollaborativeStrategy(
            target_batch_size=target_batch_size,
            grad_compression=compression,
            state_averaging_compression=compression,
        ),
        accelerator="gpu",
        devices=1,
    )


PowerSGD
""""""""

`PowerSGD <https://arxiv.org/abs/1905.13727>`_ is a technique to reduce distributed communication of gradients across processes.
In short, PowerSGD uses a low rank approximation to compress gradients before all reducing.

.. note::
    PowerSGD may impact convergence, however it is worth trying as it can substantially reduce the communication between processes.

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.strategies import CollaborativeStrategy
    from functools import partial
    from hivemind.optim.power_sgd_averager import PowerSGDGradientAverager

    trainer = pl.Trainer(
        strategy=CollaborativeStrategy(
            target_batch_size=8192,
            grad_averager_factory=partial(PowerSGDGradientAverager, averager_rank=32, min_compression_ratio=0.5),
        ),
        accelerator="gpu",
        devices=1,
    )
