.. list-table:: reg. user 1.9
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used Python 3.7
     - upgrade to Python 3.8 or higher
     - #16579

   * - used PyTorch 1.10
     - upgrade to PyTorch 1.11 or higher
     - #16492

   * - used Trainer’s flag ``gpus``
     - use ``devices`` with the same number
     - #16171

   * - used Trainer’s flag ``tpu_cores``
     - use ``devices`` with the same number
     - #16171

   * - used Trainer’s flag ``ipus``
     - use ``devices`` with the same number
     - #16171

   * - used Trainer’s flag ``num_processes``
     - use ``devices`` with the same number
     - #16171

   * - used Trainer’s flag ``resume_from_checkpoint``
     - pass the path to the ``Trainer.fit(ckpt_path="...")`` method,
     - #10061

   * - used Trainer’s flag ``auto_select_gpus``
     - use ``devices="auto"``
     - #16184

   * - called the ``pl.tuner.auto_gpu_select.pick_single_gpu`` function
     - use Trainer’s flag``devices="auto"``
     - #16184

   * - called the ``pl.tuner.auto_gpu_select.pick_multiple_gpus`` functions
     - use Trainer’s flag``devices="auto"``
     - #16184

   * - used Trainer’s flag  ``accumulate_grad_batches`` with a scheduling dictionary value
     - use the  ``GradientAccumulationScheduler`` callback and configure it
     - #16729

   * - imported profiles from ``pl.profiler``
     - import from ``pl.profilers``
     - #16359

   * - used ``Tuner`` as part of ``Trainer`` in any form
     - move to a standalone ``Tuner`` object or use particular callbacks ``LearningRateFinder`` and ``BatchSizeFinder``
     - https://lightning.ai/docs/pytorch/latest/advanced/training_tricks.html#batch-size-finder

   * - used Trainer’s flag ``auto_scale_batch_size``
     - use ``BatchSizeFinder`` callback instead and the ``Trainer.tune()`` method was removed
     -

   * - used Trainer’s flag ``auto_lr_find``
     - use callbacks ``LearningRateFinder`` callback instead and the ``Trainer.tune()`` method was removed
     -
