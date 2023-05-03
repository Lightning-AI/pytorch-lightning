.. list-table:: devel 1.8
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - derived from ``pytorch_lightning.loggers.base.LightningLoggerBase``
     - derive from ``pytorch_lightning.loggers.logger.Logger``
     - `PR12014`_

   * - derived from ``pytorch_lightning.profiler.base.BaseProfiler``
     - derive from ``pytorch_lightning.profilers.profiler.Profiler``
     - `PR12150`_

   * - derived from ``pytorch_lightning.profiler.base.AbstractProfiler``
     - derive from ``pytorch_lightning.profilers.profiler.Profiler``
     - `PR12106`_


.. _pr12014: https://github.com/Lightning-AI/lightning/pull/12014
.. _pr12150: https://github.com/Lightning-AI/lightning/pull/12150
.. _pr12106: https://github.com/Lightning-AI/lightning/pull/12106
