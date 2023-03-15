.. list-table:: adv. user 1.4
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - called ``ModelCheckpoint.save_function``
     - now call ``Trainer.save_checkpoint``
     - #7201

   * - accessed the ``Trainer.running_sanity_check`` property
     - now  access the ``Trainer.sanity_checking`` property
     - #4945

   * - used ``LightningModule.grad_norm``
     - now use the ``pl.utilities.grad_norm`` utility function instead
     - #7292

   * - used ``TrainerTrainingTricksMixin.detect_nan_tensors``
     - now use ``pl.utilities.grads.grad_norm``
     - #6834

   * - used ``TrainerTrainingTricksMixin.print_nan_gradients``
     - now use ``pl.utilities.finite_checks.print_nan_gradients``
     - #6834

   * - If you relied on ``TrainerLoggingMixin.metrics_to_scalars``
     - now use ``pl.utilities.metrics.metrics_to_scalars``
     - #7180

   * - selected the i-th GPU with ``Trainer(gpus="i,j")``
     - now this will set the number of GPUs, just like passing ``Trainer(devices=i)``, you can still select the specific GPU by setting the ``CUDA_VISIBLE_DEVICES=i,j`` environment variable
     - #6388
