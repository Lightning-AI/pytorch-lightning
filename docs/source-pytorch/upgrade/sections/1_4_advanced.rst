.. list-table:: adv. user 1.4
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - called ``ModelCheckpoint.save_function``
     - now call ``Trainer.save_checkpoint``
     - `PR7201`_

   * - accessed the ``Trainer.running_sanity_check`` property
     - now  access the ``Trainer.sanity_checking`` property
     - `PR4945`_

   * - used ``LightningModule.grad_norm``
     - now use the ``pl.utilities.grad_norm`` utility function instead
     - `PR7292`_

   * - used ``TrainerTrainingTricksMixin.detect_nan_tensors``
     - now use ``pl.utilities.grads.grad_norm``
     - `PR6834`_

   * - used ``TrainerTrainingTricksMixin.print_nan_gradients``
     - now use ``pl.utilities.finite_checks.print_nan_gradients``
     - `PR6834`_

   * - If you relied on ``TrainerLoggingMixin.metrics_to_scalars``
     - now use ``pl.utilities.metrics.metrics_to_scalars``
     - `PR7180`_

   * - selected the i-th GPU with ``Trainer(gpus="i,j")``
     - now this will set the number of GPUs, just like passing ``Trainer(devices=i)``, you can still select the specific GPU by setting the ``CUDA_VISIBLE_DEVICES=i,j`` environment variable
     - `PR6388`_


.. _pr7201: https://github.com/Lightning-AI/lightning/pull/7201
.. _pr4945: https://github.com/Lightning-AI/lightning/pull/4945
.. _pr7292: https://github.com/Lightning-AI/lightning/pull/7292
.. _pr6834: https://github.com/Lightning-AI/lightning/pull/6834
.. _pr7180: https://github.com/Lightning-AI/lightning/pull/7180
.. _pr6388: https://github.com/Lightning-AI/lightning/pull/6388
