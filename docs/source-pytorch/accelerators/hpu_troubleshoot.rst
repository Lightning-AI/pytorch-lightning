:orphan:

.. _hpu_troubleshoot:

Accelerator: HPU training
=========================
**Audience:** Gaudi chip users looking to troubleshoot their model changes during training with HPU.

----

Runtime Errors
--------------

This section provides troubleshooting instructions that can be referred to for common issues when training PyTorch models. The following are common functional issues that may occur when running on HPU and not on CPU/GPU.

The following table outlines some possible runtime errors:

+----------------------------------+----------------------------------+
| Error                            | Solution                         |
+==================================+==================================+
| The `HPUAccelerator` can only    | Ensure either of                 |
| be used with a                   | `SingleHPUStrategy` or           |
| `SingleHPUStrategy` or           | HPUParallelStrategy` is used  in |
| `HPUParallelStrategy`, found     | the Trainer class                |
| DDPFullyShardedStrategy          |                                  |
| Or any other strategy            |                                  |
+----------------------------------+----------------------------------+
| `SingleHPUStrategy` requires     | Ensure HPU backend is available  |
|  HPU devices to run              | or devices initialized           |
+----------------------------------+----------------------------------+
| `HPUParallelStrategy` requires   | Ensure HPU backend is available  |
|  HPU devices to run              | or devices initialized           |
+----------------------------------+----------------------------------+
| HPU precision plugin requires    | Ensure HPU backend is available  |
| HPU devices                      | or HPU accelerator used          |
+----------------------------------+----------------------------------+

For more details, please refer to `Troubleshooting your Model <https://docs.habana.ai/en/latest/PyTorch/Debugging_Guide/Model_Troubleshooting.html>`__.
