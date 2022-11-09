.. _api_migration:

API Migration Guide
###################


API Changes
***********

.. list-table:: API changes
   :widths: 40 20 20 20
   :header-rows: 1

   * -
     - DDP
     - DDP Spawn
     - DDP Notebook/Fork
   * - Works in Jupyter notebooks / IPython environments
     - No
     - No
     - Yes
   * - Supports multi-node
     - Yes
     - Yes
     - Yes
   * - Supported platforms
     - Linux, Mac, Win
     - Linux, Mac, Win
     - Linux, Mac
   * - Requires all objects to be picklable
     - No
     - Yes
     - No
   * - Is the guard ``if __name__=="__main__"`` required?
     - Yes
     - Yes
     - No
   * - Limitations in the main process
     - None
     - None
     - GPU operations such as moving tensors to the GPU or calling ``torch.cuda`` functions before invoking ``Trainer.fit`` is not allowed.
   * - Process creation time
     - Slow
     - Slow
     - Fast
