:orphan:

##########################
Run in the Lightning Cloud
##########################

**Audience**: The easy and hassle-free way to run large training jobs in the cloud. No infrastructure setup required.


----


*************
Initial Setup
*************

First, create a free `Lightning AI account <https://lightning.ai/>`_.
Then, log in from the CLI:

.. code-block:: bash

    lightning login

A page opens in your browser.
Follow the instructions there to complete the setup.


----


***************************************
Launch multi-node training in the cloud
***************************************

**Step 1:** Put your code inside a :class:`~lightning_app.core.work.LightningWork`:

.. code-block:: python
    :emphasize-lines: 5
    :caption: app.py

    import lightning as L
    from lightning.app.components import LiteMultiNode

    # 1. Put your code inside a LightningWork
    class MyTrainingComponent(L.LightningWork):
        def run(self):

            # Set up Fabric
            # The `devices` and `num_nodes` gets set by Lightning automatically
            fabric = L.Fabric(strategy="ddp", precision=16)

            # Your training code
            model = ...
            optimizer = ...
            model, optimizer = fabric.setup(model, optimizer)
            ...

**Step 2:** Configure the number of nodes, the number of GPUs per node, and the type of GPU:

.. code-block:: python
    :emphasize-lines: 5,7
    :caption: app.py

    # 2. Create the app with the LiteMultiNode component inside
    app = L.LightningApp(
        LiteMultiNode(
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
