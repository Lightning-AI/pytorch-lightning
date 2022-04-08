.. _collaborative_training:

Collaborative Training
======================

Collaborative Training tries to solve the need for top-tier multi-GPU servers by allowing you to train across unreliable machines,
such as local machines or even preemptible cloud compute across the internet.

Under the hood, we use `Hivemind <https://github.com/learning-at-home/hivemind>`_ which provides de-centralized training across the internet.

To use Collaborative Training, you need to first install Hivemind.

.. code-block:: bash

    pip install hivemind

The ``CollaborativeStrategy`` accumulates gradients from all processes that are collaborating till they reach a ``target_batch_size``. By default, we use the batch size
of the first batch to determine what each local machine batch contributes towards the ``target_batch_size``. Once the ``target_batch_size`` is reached, an optimizer step
is made on all processes.

.. warning::

    When using ``CollaborativeStrategy`` note that you cannot use gradient accumulation (``accumulate_grad_batches``). This is because hivemind manages accumulation internally.

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.strategies import CollaborativeStrategy


    trainer = pl.Trainer(strategy=CollaborativeStrategy(target_batch_size=8192))


.. code-block:: bash

    python train.py
    # Other machines can connect running the same command:
    # INITIAL_PEERS=... python train.py
    # or passing the peers to the strategy:"
    # CollaborativeStrategy(initial_peers=...)"

Once training starts a helper message is printed, showing how to start training on other machines using the same code.

.. _collaborative_training_optimization:

Reducing Communication Bottlenecks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below are some ways to reduce communication when training collaboratively. As the model sizes get larger, bottlenecks in communication become more apparent.

Overlap Gradient and State Averaging
""""""""""""""""""""""""""""""""""""

When the target batch size is reached, all processes that are included in the step send gradients and model states to each other. By enabling some flags through
the strategy, communication can happen in the background. This allows training to continue (with slightly outdated weights) but provides us the means
to overlap communication with computation.

.. warning::
    Enabling these flags means that you must pass in a ``scheduler_fn`` to the ``CollaborativeStrategy`` instead of relying on a scheduler from ``configure_optimizers``.
    The optimizer is re-created by Hivemind, and as a result the scheduler has to as well.

.. code-block:: python

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
            scheduler_fn=partial(MyLearningRateScheduler, num_warmup_steps=..., num_training_steps=...),
        ),
        gpus=1,
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
    )
