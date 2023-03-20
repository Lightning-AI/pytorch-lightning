.. list-table:: devel 1.5
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref


   * - called ``CheckpointConnector.hpc_load()``
     - just call ``CheckpointConnector.restore()``
     - #7652

   * - used ``TrainerModelHooksMixin``
     - now rely on the corresponding utility functions in ``pytorch_lightning.utilities.signature_utils``
     - #7422

   * - assigned the ``Trainer.train_loop`` property
     - now assign the equivalent ``Trainer.fit_loop`` property
     - #8025

   * - accessed ``LightningModule.loaded_optimizer_states_dict``
     - the property has been removed
     - #8229
