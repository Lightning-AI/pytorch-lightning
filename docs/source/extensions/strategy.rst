.. _strategy:

########
Strategy
########

Strategy depicts the training strategy to be used by the :doc:`Lightning Trainer <../common/trainer>`. It can be controlled by passing different
training strategies with aliases (``ddp``, ``ddp_spawn``, etc) as well as custom training strategies to the ``strategy`` parameter for Trainer.

.. code-block:: python

    # Training with the DistributedDataParallel strategy on 4 gpus
    trainer = Trainer(strategy="ddp", accelerator="gpu", devices=4)

    # Training with the custom DistributedDataParallel strategy on 4 gpus
    trainer = Trainer(strategy=DDPPlugin(...), accelerator="gpu", devices=4)

    # Training with the DDP Spawn strategy using auto accelerator selection
    trainer = Trainer(strategy="ddp_spawn", accelerator="auto", devices=4)

    # Training with the DeepSpeed strategy on available gpus
    trainer = Trainer(strategy="deepspeed", accelerator="gpu", devices="auto")

    # Training with the DDP strategy using 3 cpu processes
    trainer = Trainer(strategy="ddp", accelerator="cpu", devices=3)

    # Training with the DDP Spawn strategy on 8 tpu cores
    trainer = Trainer(strategy="ddp_spawn", accelerator="tpu", devices=8)

    # Training with the default IPU strategy on 8 ipus
    trainer = Trainer(accelerator="ipu", devices=8)
