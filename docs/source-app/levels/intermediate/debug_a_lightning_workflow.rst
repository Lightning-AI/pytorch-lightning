###################################
Level 3: Debug A Lightning Workflow
###################################
**Audience:** Users who want to debug Lightning workflows.

**Prereqs:** You must have finished the `Basic levels <https://lightning.ai/lightning-docs/levels/basic/>`_.

----

****************
The Toy workflow
****************
In this page, we'll be using the following toy snippet:

.. code:: python

   # app.py
   import lightning as L

   class LitWorker(L.LightningWork):
        def run(self):
            print("run ANY python code here")

    app = L.LightningApp(LitWorker())

----

*************
Debug locally
*************
Lightning runs distributed cloud workflows locally for development and debugging purposes.
Enable debugging by using **MultiProcessRuntime**:

.. code:: python
   :emphasize-lines: 3, 10

   # app.py
   import lightning as L
   from lightning.app.runners import MultiProcessRuntime

   class LitWorker(L.LightningWork):
        def run(self):
            print("run ANY python code here")

    app = L.LightningApp(LitWorker())
    MultiProcessRuntime(app).dispatch()
