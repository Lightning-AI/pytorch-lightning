.. list-table:: adv. user 1.8
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - imported ``pl.callbacks.base``
     - import ``pl.callbacks.callback``
     - #13031

   * - imported ``pl.loops.base``
     - import ``pl.loops.loop`` instead
     - #13043

   * - imported ``pl.utilities.cli``
     - import  ``pl.cli`` instead
     - #13767

   * - imported profiler classes from ``pl.profiler.*``
     - import ``pl.profilers`` instead
     - #12308

   * - used ``pl.accelerators.GPUAccelerator``
     - use ``pl.accelerators.CUDAAccelerator``
     - #13636

   * - used ``LightningDeepSpeedModule``
     - use ``strategy="deepspeed"`` or ``strategy=DeepSpeedStrategy(...)``
     - https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.strategies.DeepSpeedStrategy.html

   * - used the ``with init_meta_context()`` context manager from ``import pl.utilities.meta``
     - switch to ``deepspeed-zero-stage-3``
     - https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed-zero-stage-3

   * - used the Lightning Hydra multi-run integration
     - removed support for it as it caused issues with processes hanging
     - #15689

   * - used ``pl.utilities.memory.get_gpu_memory_map``
     - use  ``pl.accelerators.cuda.get_nvidia_gpu_stats``
     - #9921