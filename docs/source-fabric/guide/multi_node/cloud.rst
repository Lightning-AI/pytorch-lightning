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
You get free credits every month you can spend on GPU compute.
To use muliple machines, you need to be on the `Pro or Teams plan <https://lightning.ai/pricing>`_.


----


***************************************
Launch multi-node training in the cloud
***************************************

**Step 1:** Start a new studio.
**Step 2:** Bring your code into the studio. You can clone a GitHub repo, drag and drop local files, or use the following demo example:

.. collapse:: Example

    .. code-block:: python
        :caption: main.py

        # TODO

**Step 3:** Remove any hardcoded accelerator settings and let Lightning automatically set them for you. **No other changes are required in your script.**

.. code-block:: python
    :caption: main.py

    # These are the defaults
    fabric = L.Fabric(accelerator="auto", devices="auto")

    # DON'T hardcode these, leave them default/auto
    # fabric = L.Fabric(accelerator="cpu", devices=3)


**Step 4:** Install dependencies and download all necessary data. Test that your script runs in the studio first. **If it runs in the studio, it will run in multi-node!**


**Step 5:** Open the Multi-Machine Training (MMT) app. Type the command to run your script, select the machine type and how many machines you want to launch on. Click "Run" to start the job.


----


****************************
Bring your own cloud account
****************************

On the `Teams or Enterprise <https://lightning.ai/pricing>`_ tier, you can connect your own AWS account.



----


**********
Next steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Lightning Platform
    :description: Develop, Train and Deploy models on the cloud
    :col_css: col-md-4
    :button_link: https://lightning.ai
    :height: 150

.. raw:: html

        </div>
    </div>
