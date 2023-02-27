:orphan:

##########################
Run in the Lightning Cloud
##########################

**Audience**: Users who don't want to waste time on cluster configuration and maintenance.


The Lightning AI cloud is a platform where you can build, train, finetune and deploy models without worrying about infrastructure, cost management, scaling, and other technical headaches.
In this guide, and within just 10 minutes, you will learn how to run a Fabric training script across multiple nodes in the cloud.


----


*************
Initial Setup
*************

First, create a free `Lightning AI account <https://lightning.ai/>`_.
Then, log in from the CLI:

.. code-block:: bash

    lightning login

A page opens in your browser where you can follow the instructions to complete the setup.


----


***************************************
Launch multi-node training in the cloud
***************************************

**Step 1:** Put your code inside a :class:`~lightning_app.core.work.LightningWork`:

.. code-block:: python
    :emphasize-lines: 5
    :caption: app.py

    import lightning as L
    from lightning.app.components import FabricMultiNode


    # 1. Put your code inside a LightningWork
    class MyTrainingComponent(L.LightningWork):
        def run(self):
            # Set up Fabric
            # The `devices` and `num_nodes` gets set by Lightning automatically
            fabric = L.Fabric(strategy="ddp", precision="16-mixed")

            # Your training code
            model = ...
            optimizer = ...
            model, optimizer = fabric.setup(model, optimizer)
            ...

**Step 2:** Init a :class:`~lightning_app.core.app.LightningApp` with the ``FabricMultiNode`` component.
Configure the number of nodes, the number of GPUs per node, and the type of GPU:

.. code-block:: python
    :emphasize-lines: 5,7
    :caption: app.py

    # 2. Create the app with the FabricMultiNode component inside
    app = L.LightningApp(
        FabricMultiNode(
            MyTrainingComponent,
            # Run with 2 nodes
            num_nodes=2,
            # Each with 4 x V100 GPUs, total 8 GPUs
            cloud_compute=L.CloudCompute("gpu-fast-multi"),
        )
    )


**Step 3:** Run your code from the CLI:

.. code-block:: bash

    lightning run app app.py --cloud

This command will upload your Python file and then opens the app admin view, where you can see the logs of what's happening.

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric/fabric-multi-node-admin.png
   :alt: The Lightning AI admin page of an app running a multi-node fabric training script
   :width: 100%


----


**********
Next steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Lightning App Docs
    :description: Learn more about apps and the Lightning cloud.
    :col_css: col-md-4
    :button_link: https://lightning.ai
    :height: 150

.. raw:: html

        </div>
    </div>
