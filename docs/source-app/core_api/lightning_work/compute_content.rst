
***************************
Customize my Work resources
***************************

In the cloud, you can simply configure which machine to run on by passing
a :class:`~lightning_app.utilities.packaging.cloud_compute.CloudCompute` to your work ``__init__`` method:

.. code-block:: python

    import lightning as L

    # Run on a free, shared CPU machine. This is the default for every LightningWork.
    MyCustomWork(cloud_compute=L.CloudCompute())

    # Run on a dedicated, medium-size CPU machine (see specs below)
    MyCustomWork(cloud_compute=L.CloudCompute("cpu-medium"))

    # Run on cheap GPU machine with a single GPU (see specs below)
    MyCustomWork(cloud_compute=L.CloudCompute("gpu"))

    # Run on a fast multi-GPU machine (see specs below)
    MyCustomWork(cloud_compute=L.CloudCompute("gpu-fast-multi"))


Here is the full list of supported machine names:

.. list-table:: Hardware by Accelerator Type
   :widths: 25 25 25 25
   :header-rows: 1

   * - Name
     - # of CPUs
     - GPUs
     - Memory
   * - default
     - 1
     - 0
     - 4 GB
   * - cpu-small
     - 2
     - 0
     - 8 GB
   * - cpu-medium
     - 8
     - 0
     - 32 GB
   * - gpu
     - 4
     - 1 (T4, 16 GB)
     - 16 GB
   * - gpu-fast
     - 8
     - 1 (V100, 16 GB)
     - 61 GB
   * - gpu-fast-multi
     - 32
     - 4 (V100 16 GB)
     - 244 GB

To start running a Lightning app on the cloud, you will need to verify your account has sufficient credits. Read :ref:`credits` to learn more. The up-to-date prices for these instances can be found `here <https://lightning.ai/pages/pricing>`_.


.. note::

    When running a Lighting App on your local machine, any :class:`~lightning_app.utilities.packaging.cloud_compute.CloudCompute`
    configuration (including a :class:`~lightning_app.storage.mount.Mount`) is ignored at runtime. If you need access to
    these files on your local disk, you should download a copy of them to your machine.

----

************
CloudCompute
************

.. autoclass:: lightning_app.utilities.packaging.cloud_compute.CloudCompute
    :noindex:
