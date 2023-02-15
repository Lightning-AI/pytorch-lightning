:orphan:

***************************
Customize my Flow resources
***************************

In the cloud, you can simply configure which machine to run on by passing
a :class:`~lightning.app.utilities.packaging.cloud_compute.CloudCompute` to your work ``__init__`` method:

.. code-block:: python

    import lightning as L

    # Run on a small, shared CPU machine. This is the default for every LightningFlow.
    app = L.LightningApp(L.Flow(), flow_cloud_compute=L.CloudCompute())


Here is the full list of supported machine names:

.. list-table:: Hardware by Accelerator Type
   :widths: 25 25 25
   :header-rows: 1

   * - Name
     - # of CPUs
     - Memory
   * - flow-lite
     - 0.3
     - 4 GB

The up-to-date prices for these instances can be found `here <https://lightning.ai/pages/pricing>`_.

----

************
CloudCompute
************

.. autoclass:: lightning.app.utilities.packaging.cloud_compute.CloudCompute
    :noindex:
