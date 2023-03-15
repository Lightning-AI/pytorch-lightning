.. list-table:: reg. user 1.9
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - PyTorch 1.10
     - upgrade to min PyTorch 1.11
     - #16492

   * - Python 3.7
     - upgrade to min Python 3.7
     - #16579

   * - used Trainer’s flag ``gpus``
     - use ``devices`` with the same value
     - #16171

   * - used Trainer’s flag ``tpu_cores``
     - use ``devices`` with the same value
     - #16171

   * - used Trainer’s flag ``ipus``
     - use ``devices`` with the same value
     - #16171

   * - used Trainer’s flag ``num_processes``
     - use ``devices`` with the same value
     - #16171

   * - used Trainer’s flag ``resume_from_checkpoint``
     - pass this path to the fit functions instead, for example, ``trainer.fit(ckpt_path="...")``
     - #10061

   * - used Trainer’s flag ``auto_select_gpus``
     - now use ``devices="auto"``
     - #16184

   * - called the ``pl.tuner.auto_gpu_select.pick_single_gpu`` or ``pl.tuner.auto_gpu_select.pick_multiple_gpus`` functions
     - now use Trainer argument``devices="auto"``
     - #16184

   * - used Trainer’s flag  ``accumulate_grad_batches`` with a scheduling dictionary value
     - use the  ``GradientAccumulationScheduler`` callback
     - #16729

   * - were importing profiles from ``pl.profiler``
     - import from ``pl.profilers``
     - #16359

   * - have implemented ``training_epoch_end`` hooks
     - port your logic to  ``on_training_epoch_end`` hook instead
     - #16520

   * - have implemented ``validation_epoch_end`` hook
     - port your logic to  ``on_validation_epoch_end`` hook instead
     - #16520

   * - have implemented ``test_epoch_end`` hooks
     - port your logic to  ``on_test_epoch_end`` hook instead
     - #16520

   * - used Tuner as part of Trainer in any form
     - move to a standalone tuner object or use particular callbacks ``LearningRateFinder`` and ``BatchSizeFinder`` instead
     - https://lightning.ai/docs/pytorch/latest/advanced/training_tricks.html#batch-size-finder

   * - used Trainer’s flag ``auto_scale_batch_size``
     - use ``BatchSizeFinder`` callback instead and the ``Trainer.tune()`` method was removed
     - ...

   * - used Trainer’s flag ``auto_lr_find``
     - use callbacks ``LearningRateFinder`` callback instead and the ``Trainer.tune() `` method was removed
     - ...