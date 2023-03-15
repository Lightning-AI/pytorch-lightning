.. list-table:: devel 1.6
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - called ``LightningLoggerBase.close``
     - switch to ``LightningLoggerBase.finalize``.
     - #9422

   * - called ``LoggerCollection.close``
     - switch to ``LoggerCollection.finalize``.
     - #9422

   * - used ``AcceleratorConnector.is_slurm_managing_tasks`` attribute
     - it is set not as protected and discouraged from direct use
     - #10101

   * - used ``AcceleratorConnector.configure_slurm_ddp`` attributes
     - it is set not as protected and discouraged from direct use
     - #10101

   * - used ``ClusterEnvironment.creates_children()`` method
     - change it to ``ClusterEnvironment.creates_processes_externally`` which is property now.
     - #10106

   * - called ``PrecisionPlugin.master_params()``
     - update it  ``PrecisionPlugin.main_params()``
     - #10105
