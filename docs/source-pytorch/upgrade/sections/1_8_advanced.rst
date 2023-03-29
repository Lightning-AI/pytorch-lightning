.. list-table:: adv. user 1.8
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - imported ``pl.callbacks.base``
     - import ``pl.callbacks.callback``
     - `PR13031`_

   * - imported ``pl.loops.base``
     - import ``pl.loops.loop`` instead
     - `PR13043`_

   * - imported ``pl.utilities.cli``
     - import  ``pl.cli`` instead
     - `PR13767`_

   * - imported profiler classes from ``pl.profiler.*``
     - import ``pl.profilers`` instead
     - `PR12308`_

   * - used ``pl.accelerators.GPUAccelerator``
     - use ``pl.accelerators.CUDAAccelerator``
     - `PR13636`_

   * - used ``LightningDeepSpeedModule``
     - use ``strategy="deepspeed"`` or ``strategy=DeepSpeedStrategy(...)``
     - :class:`~lightning.pytorch.strategies.DeepSpeedStrategy`

   * - used the ``with init_meta_context()`` context manager from ``import pl.utilities.meta``
     - switch to ``deepspeed-zero-stage-3``
     - :ref:`deepspeed-zero-stage-3`

   * - used the Lightning Hydra multi-run integration
     - removed support for it as it caused issues with processes hanging
     - `PR15689`_

   * - used ``pl.utilities.memory.get_gpu_memory_map``
     - use  ``pl.accelerators.cuda.get_nvidia_gpu_stats``
     - `PR9921`_


.. _pr13031: https://github.com/Lightning-AI/lightning/pull/13031
.. _pr13043: https://github.com/Lightning-AI/lightning/pull/13043
.. _pr13767: https://github.com/Lightning-AI/lightning/pull/13767
.. _pr12308: https://github.com/Lightning-AI/lightning/pull/12308
.. _pr13636: https://github.com/Lightning-AI/lightning/pull/13636
.. _pr15689: https://github.com/Lightning-AI/lightning/pull/15689
.. _pr9921: https://github.com/Lightning-AI/lightning/pull/9921
