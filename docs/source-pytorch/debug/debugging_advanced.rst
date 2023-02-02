:orphan:

.. _debugging_advanced:

###########################
Debug your model (advanced)
###########################
**Audience**: Users who want to debug distributed models.

----

************************
Debug distributed models
************************
To debug a distributed model, we recommend you debug it locally by running the distributed version on CPUs:

.. code-block:: python

    trainer = Trainer(accelerator="cpu", strategy="ddp", devices=2)

On the CPU, you can use `pdb <https://docs.python.org/3/library/pdb.html>`_ or `breakpoint() <https://docs.python.org/3/library/functions.html#breakpoint>`_
or use regular print statements.

.. testcode::

    class LitModel(LightningModule):
        def training_step(self, batch, batch_idx):
            debugging_message = ...
            print(f"RANK - {self.trainer.global_rank}: {debugging_message}")

            if self.trainer.global_rank == 0:
                import pdb

                pdb.set_trace()

            # to prevent other processes from moving forward until all processes are in sync
            self.trainer.strategy.barrier()

When everything works, switch back to GPU by changing only the accelerator.

.. code-block:: python

    trainer = Trainer(accelerator="gpu", strategy="ddp", devices=2)
