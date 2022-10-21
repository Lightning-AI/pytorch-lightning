:orphan:

############################
Checkpointing (early access)
############################

Lightning app checkpointing makes it easy to start save the state of a lightning app and start it from a saved checkpoint.

----

*********************
What is a checkpoint?
*********************
When a lightning app is running it is possible to save the state of the app and start it from a saved checkpoint.
This is useful for long running apps that need to be stopped and restarted or while developing an app and needing to recreate the state of the app to to continue development at a specific point.


************************
How to save a checkpoint
************************

Saving a checkpoint is simple. In main flow if your lightning app, implement `should_save_checkpoint()` that returns True when you want to save a checkpoint.

.. code-block:: python

    class MyLightningApp(L.LightningFlow):
        def run(self):
            # do stuff
            pass

        def should_save_checkpoint(self):
            return True

When `should_save_checkpoint()` returns True, Lightning will save the state of the app to a checkpoint file. The checkpoint file will be saved in lightning shared storage at `/checkpoints` and will be named `lightningapp_checkpoint_<timestamp>.json`.


*************************************
How to an app start from a checkpoint
*************************************


To start from a checkpoint, use the `--checkpoint` argument when starting the app. The value of the argument should be either:
1. The path to the local checkpoint file


.. code-block:: bash

    lightning run app app.py --checkpoint lightningapp_checkpoint_1665501626.json


2. The name of the checkpoint file in lightning shared storage
.. code-block:: bash

    lightning run app app.py --checkpoint lightningapp_checkpoint_1665501688


3. keyword `latest` to start from the latest checkpoint file in lightning shared storage.
.. code-block:: bash

    lightning run app app.py --checkpoint latest

When starting from a checkpoint, Lightning will load the checkpoint file and start the app from the saved state and update all app components with the saved state.


***************************
Checkpointing compatibility
***************************

When starting an app from a saved checkpoint, it has to be compatible with the app code.
This means that the app code has to be able to load the saved state and update all app components with the saved state.
If the checkpoint contains a state of a component that is not in the app code, the app won't start and will raise an exception.

To control how the checkpoint is loaded and create the missing components, implement `load_state_dict()` in your app. You can also implement any migration logic needed to read or update the checkpoint before loading it.


.. code-block:: python

    class Work(L.LightningWork):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def run(self):
            self.counter += 1


    class MyLightningApp(L.LightningFlow):
        def run(self):
            # dynamically create a work.
            if not getattr(self, "w", None):
                self.w = WorkReload()

            self.w.run()

        def load_state_dict(self, flow_state, children_states, strict) -> None:
            # 1: Re-instantiate the dynamic work component
            self.w = Work()

            # 2: Make any states modification / migration.
            ...

            # 3: Call the parent ``load_state_dict`` to
            # recursively reload the states.
            super().load_state_dict(
                flow_state,
                children_states,
                strict,
            )


If you see this exception "The component <component_name> wasn't instantiated for the component root", it means that the checkpoint is not compatible with the app code and you need to implement `load_state_dict()` and make sure that all components in the checkpoint are instantiated.