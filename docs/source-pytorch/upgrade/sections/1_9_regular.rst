.. list-table:: reg. user 1.9
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used Python 3.7
     - upgrade to Python 3.8 or higher
     - `PR16579`_

   * - used PyTorch 1.10
     - upgrade to PyTorch 1.11 or higher
     - `PR16492`_

   * - used Trainer’s flag ``gpus``
     - use ``devices`` with the same number
     - `PR16171`_

   * - used Trainer’s flag ``tpu_cores``
     - use ``devices`` with the same number
     - `PR16171`_

   * - used Trainer’s flag ``ipus``
     - use ``devices`` with the same number
     - `PR16171`_

   * - used Trainer’s flag ``num_processes``
     - use ``devices`` with the same number
     - `PR16171`_

   * - used Trainer’s flag ``resume_from_checkpoint``
     - pass the path to the ``Trainer.fit(ckpt_path="...")`` method,
     - `PR10061`_

   * - used Trainer’s flag ``auto_select_gpus``
     - use ``devices="auto"``
     - `PR16184`_

   * - called the ``pl.tuner.auto_gpu_select.pick_single_gpu`` function
     - use Trainer’s flag ``devices="auto"``
     - `PR16184`_

   * - called the ``pl.tuner.auto_gpu_select.pick_multiple_gpus`` functions
     - use Trainer’s flag ``devices="auto"``
     - `PR16184`_

   * - used Trainer’s flag  ``accumulate_grad_batches`` with a scheduling dictionary value
     - use the  ``GradientAccumulationScheduler`` callback and configure it
     - `PR16729`_

   * - imported profiles from ``pl.profiler``
     - import from ``pl.profilers``
     - `PR16359`_

   * - used ``Tuner`` as part of ``Trainer`` in any form
     - move to a standalone ``Tuner`` object or use particular callbacks ``LearningRateFinder`` and ``BatchSizeFinder``
     - :ref:`batch_size_finder` :ref:`learning_rate_finder`

   * - used Trainer’s flag ``auto_scale_batch_size``
     - use ``BatchSizeFinder`` callback instead and the ``Trainer.tune()`` method was removed
     -

   * - used Trainer’s flag ``auto_lr_find``
     - use callbacks ``LearningRateFinder`` callback instead and the ``Trainer.tune()`` method was removed
     -

.. _pr16579: https://github.com/Lightning-AI/lightning/pull/16579
.. _pr16492: https://github.com/Lightning-AI/lightning/pull/16492
.. _pr10061: https://github.com/Lightning-AI/lightning/pull/10061
.. _pr16171: https://github.com/Lightning-AI/lightning/pull/16171
.. _pr16184: https://github.com/Lightning-AI/lightning/pull/16184
.. _pr16729: https://github.com/Lightning-AI/lightning/pull/16729
.. _pr16359: https://github.com/Lightning-AI/lightning/pull/16359
