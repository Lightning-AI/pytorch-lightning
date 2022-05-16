:orphan:

.. _collaborative_training_intermediate:

Training on unreliable mixed GPUs across the internet (Intermediate)
====================================================================

Reducing Communication By Overlapping Communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can reduce the impact of communication across all machines by overlapping communication with our training iterations. In short, we enable communication to happen
in the background of training.

Overlap Gradient and State Averaging
""""""""""""""""""""""""""""""""""""

When the target batch size is reached, all processes that are included in the step send gradients and model states to each other. By enabling some flags through
the strategy, communication can happen in the background. This allows training to continue (with slightly outdated weights) but provides us the means
to overlap communication with computation.

.. warning::
    Enabling overlapping communication means convergence will slightly be affected.

.. note::
    Enabling these flags means that you must pass in a ``scheduler_fn`` to the ``CollaborativeStrategy`` instead of relying on a scheduler from ``configure_optimizers``.
    The optimizer is re-created by Hivemind, and as a result, the scheduler has to be re-created.

.. code-block:: python

    import torch
    from functools import partial
    import pytorch_lightning as pl
    from pytorch_lightning.strategies import CollaborativeStrategy

    trainer = pl.Trainer(
        strategy=CollaborativeStrategy(
            target_batch_size=8192,
            delay_state_averaging=True,
            delay_grad_averaging=True,
            delay_optimizer_step=True,
            offload_optimizer=True,  # required to delay averaging
            scheduler_fn=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=...),
        ),
        accelerator="gpu",
        devices=1,
    )


Reducing GPU Memory requirements by re-using buffers & CPU offloading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can also offload the optimizer state to the CPU whilst re-using gradient buffers to reduce the memory requirement for machines.

Offloading Optimizer State to the CPU
"""""""""""""""""""""""""""""""""""""

Offloading the Optimizer state to the CPU works the same as :ref:`deepspeed-zero-stage-2-offload`, where we save GPU memory by keeping all optimizer states on the CPU.

.. note::
    Enabling these flags means that you must pass in a ``scheduler_fn`` to the ``CollaborativeStrategy`` instead of relying on a scheduler from ``configure_optimizers``.
    The optimizer is re-created by Hivemind, and as a result, the scheduler has to be re-created.

    We suggest enabling offloading and overlapping communication to hide the additional overhead from having to communicate with the CPU.

.. code-block:: python

    import torch
    from functools import partial
    import pytorch_lightning as pl
    from pytorch_lightning.strategies import CollaborativeStrategy

    trainer = pl.Trainer(
        strategy=CollaborativeStrategy(
            target_batch_size=8192,
            offload_optimizer=True,
            scheduler_fn=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=...),
        ),
        accelerator="gpu",
        devices=1,
    )


Re-using Gradient Buffers
"""""""""""""""""""""""""

By default, Hivemind accumulates gradients in a separate buffer. This means additional GPU memory is required to store gradients. You can enable re-using the model parameter gradient buffers by passing ``reuse_grad_buffers=True`` to the ``CollaborativeStrategy``.

.. warning::
    The ``CollaborativeStrategy`` will override ``zero_grad`` in your ``LightningModule`` to have no effect. This is because gradients are accumulated in the model
    and Hivemind manages when they need to be cleared.

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.strategies import CollaborativeStrategy

    trainer = pl.Trainer(
        strategy=CollaborativeStrategy(target_batch_size=8192, reuse_grad_buffers=True), accelerator="gpu", devices=1
    )
