.. list-table:: reg. user 2.0
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used PyTorch 3.11
     - upgrade to PyTorch 2.1 or higher
     - `PR18691`_

   * - called ``self.trainer.model.parameters()`` in ``LightningModule.configure_optimizers()`` when using FSDP
     - On PyTorch 2.0+, call ``self.parameters()`` from now on
     - `PR17309`_

   * - used ``Trainer(accelerator="tpu", devices=[i])"`` to select the 1-based TPU core index
     - the index is now 0-based
     - `PR17227`_

   * - used ``torch_xla < 1.13``
     - upgrade to ``torch_xla >= 1.13``
     - `PR17368`_

   * - used ``trainer.num_val_batches`` to get the total size of all validation dataloaders
     - use ``sum(trainer.num_val_batches)``
     - `PR18441`_

   * - used ``trainer.num_test_batches`` to get the total size of all test dataloaders
     - use ``sum(trainer.num_test_batches)``
     - `PR18441`_

   * - used ``trainer.num_sanity_val_batches`` to get the total size of all validation dataloaders for sanity checking
     - use ``sum(trainer.num_sanity_val_batches)``
     - `PR18441`_

   * - used ``Trainer(devices="auto")`` to auto-select all available GPUs in a Jupyter notebook
     - use ``Trainer(devices=-1)``
     - `PR18291`_

   * - used ``Trainer(devices="auto")`` to auto-select all available GPUs in a Jupyter notebook
     - use ``Trainer(devices=-1)``
     - `PR18291`_

   * - ``pip install lightning`` to install ``lightning.app`` dependencies
     - use ``pip install lightning[app]`` if you need ``lightning.app``
     - `PR18386`_


.. _pr18691: https://github.com/Lightning-AI/lightning/pull/18691
.. _pr16579: https://github.com/Lightning-AI/lightning/pull/16579
.. _pr17309: https://github.com/Lightning-AI/lightning/pull/17309
.. _pr17227: https://github.com/Lightning-AI/lightning/pull/17227
.. _pr17368: https://github.com/Lightning-AI/lightning/pull/17368
.. _pr18441: https://github.com/Lightning-AI/lightning/pull/18441
.. _pr18291: https://github.com/Lightning-AI/lightning/pull/18291
.. _pr18386: https://github.com/Lightning-AI/lightning/pull/18386
