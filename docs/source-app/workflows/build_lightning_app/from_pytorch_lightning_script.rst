#######################################################
Develop a Lightning App from a PyTorch Lightning script
#######################################################

**Audience:** Users who want to develop a Lightning App (App) from their PyTorch Lightning (PL) scripts.

----

*************************************************************
What developing a Lightning App from a PL script does for you
*************************************************************

Developing an App from a PL script allows you to immediately run on the cloud and share the progress with friends.
Once you're happy with your model, you can immediately expand beyond just model development to things like
making your own inference APIs, research demos, or even speeding up your data pipeline.

The PyTorch Lightning App is your entry point to the full end-to-end ML licefycle.

----

******************
Develop a template
******************

To develop a template from a PyTorch Lightning script, use this command:

.. code:: bash

    lightning_app init pl-app path/to/the/pl_script.py


If your script is not at the root of the project folder, and you'd like to include all source files within that folder, you can specify the root path as the first argument:

.. code:: bash

    lightning_app init pl-app path/to/project/root path/to/the/pl_script.py


The default trainer App lets you train a model with a beautiful UI locally and on the cloud with zero effort!

----

***********
Run the App
***********

.. note:: This section is under construction.

Run the App locally:

.. code:: bash

    lightning_app run app pl-app/app.py

Or run the App on the cloud so you can share with collaborators and even use all the cloud GPUs you want.

.. code:: bash

    lightning_app run app pl-app/app.py --cloud


.. figure:: https://storage.googleapis.com/grid-packages/pytorch-lightning-app/docs-thumbnail.png
    :alt: Screenshot of the PyTorch Lightning app running in the cloud


----

*******************
Modify the template
*******************

The command above generates an App file like this:

.. TODO:: list the file and show how to extend it

.. code:: python

    from your_app_name import ComponentA, ComponentB

    import lightning as L


    class LitApp(L.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.component_a = ComponentA()
            self.component_b = ComponentB()

        def run(self):
            self.component_a.run()
            self.component_b.run()


    app = L.LightningApp(LitApp())

Now you can add your own components as you wish!

----

************
Known issues
************

- The UI takes a couple seconds to load when opening the App, so please be patient.
- The timer resets when refreshing the page.
- The UI for adding new environment variables does not provide an option to delete an entry.
- A bug exists that leaves the script hanging at the start of training when using the DDP strategy.
- DDP-spawn is not supported due to pickling issues.
- It is currently not possible to submit a new run once the script has finished or failed.
