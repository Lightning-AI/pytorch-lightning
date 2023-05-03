.. list-table:: devel 1.5
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref


   * - called ``CheckpointConnector.hpc_load()``
     - just call ``CheckpointConnector.restore()``
     - `PR7652`_

   * - used ``TrainerModelHooksMixin``
     - now rely on the corresponding utility functions in ``pytorch_lightning.utilities.signature_utils``
     - `PR7422`_

   * - assigned the ``Trainer.train_loop`` property
     - now assign the equivalent ``Trainer.fit_loop`` property
     - `PR8025`_

   * - accessed ``LightningModule.loaded_optimizer_states_dict``
     - the property has been removed
     - `PR8229`_


.. _pr7652: https://github.com/Lightning-AI/lightning/pull/7652
.. _pr7422: https://github.com/Lightning-AI/lightning/pull/7422
.. _pr8025: https://github.com/Lightning-AI/lightning/pull/8025
.. _pr8229: https://github.com/Lightning-AI/lightning/pull/8229
