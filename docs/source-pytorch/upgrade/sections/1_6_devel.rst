.. list-table:: devel 1.6
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - called ``LightningLoggerBase.close``
     - switch to ``LightningLoggerBase.finalize``.
     - `PR9422`_

   * - called ``LoggerCollection.close``
     - switch to ``LoggerCollection.finalize``.
     - `PR9422`_

   * - used ``AcceleratorConnector.is_slurm_managing_tasks`` attribute
     - it is set not as protected and discouraged from direct use
     - `PR10101`_

   * - used ``AcceleratorConnector.configure_slurm_ddp`` attributes
     - it is set not as protected and discouraged from direct use
     - `PR10101`_

   * - used ``ClusterEnvironment.creates_children()`` method
     - change it to ``ClusterEnvironment.creates_processes_externally`` which is property now.
     - `PR10106`_

   * - called ``PrecisionPlugin.master_params()``
     - update it  ``PrecisionPlugin.main_params()``
     - `PR10105`_


.. _pr9422: https://github.com/Lightning-AI/lightning/pull/9422
.. _pr10101: https://github.com/Lightning-AI/lightning/pull/10101
.. _pr10105: https://github.com/Lightning-AI/lightning/pull/10105
.. _pr10106: https://github.com/Lightning-AI/lightning/pull/10106
