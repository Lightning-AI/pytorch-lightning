:orphan:

.. _collaborative_training_basic:

Training on unreliable mixed GPUs across the internet (Basic)
=============================================================

Collaborative Training tries to solve the need for top-tier multi-GPU servers by allowing you to train across unreliable machines,
such as local machines or even preemptible cloud compute across the internet.

Under the hood, we use `Hivemind <https://github.com/learning-at-home/hivemind>`_ which provides de-centralized training across the internet.

To use Collaborative Training, you need to first install Hivemind.

.. code-block:: bash

    pip install hivemind

The ``CollaborativeStrategy`` accumulates gradients from all processes that are collaborating until they reach a ``target_batch_size``. By default, we use the batch size
of the first batch to determine what each local machine batch contributes towards the ``target_batch_size``. Once the ``target_batch_size`` is reached, an optimizer step
is made on all processes.

.. warning::

    When using ``CollaborativeStrategy`` note that you cannot use gradient accumulation (``accumulate_grad_batches``). This is because Hivemind manages accumulation internally.

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.strategies import CollaborativeStrategy

    trainer = pl.Trainer(strategy=CollaborativeStrategy(target_batch_size=8192), accelerator="gpu", devices=1)

.. code-block:: bash

    python train.py
    # Other machines can connect running the same command:
    # INITIAL_PEERS=... python train.py
    # or passing the peers to the strategy:"
    # CollaborativeStrategy(initial_peers=...)"


A helper message is printed once your training begins, which shows you how to start training on other machines using the same code.
