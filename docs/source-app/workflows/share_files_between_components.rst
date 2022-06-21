:orphan:

##############################
Share Files Between Components
##############################

.. note:: The contents of this page is still in progress!

**Audience:** Users who want to share files between components.

----

**********************************
Why do I need distributed storage?
**********************************
In a Lightning App some components can be executed on their own hardware. Distributed storage
enables a file saved by a component on one machine to be used by components in other machines (transparently).

If you've asked the question "how do I use the checkpoint from this model to deploy this other thing", you've
needed distributed storage.

----

************
Write a file
************
To write a file, first create a reference to the file with the :class:`~lightning_app.storage.Path` class, then write to it:

.. code:: python

    from lightning_app.storage.path import Path

    # file reference
    boring_file_reference = Path("boring_file.txt")

    # write to that file
    with open(self.boring_file_reference, "w") as f:
        f.write("yolo")


----

**********
Use a file
**********
To use a file, pass the reference to the file:

.. code:: python

    f = open(boring_file_reference, "r")
    print(f.read())

----

..
    ********************************
    Create a directory - coming soon
    ********************************


    ----

    ******************************
    Use a directory  - coming soon
    ***************
    TODO

    ----

*********************************
Example: Share a model checkpoint
*********************************
A common workflow in ML is to use a checkpoint created by another component.
First, define a component that saves a checkpoint:

.. code:: python

    import lightning_app as lalit
    from lightning_app.storage.path import Path
    import torch
    import os


    class ModelTraining(lit.LightningWork):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_checkpoints_path = Path("/checkpoints")

        def run(self):
            # make fake checkpoints
            checkpoint_1 = torch.tensor([0, 1, 2, 3, 4])
            checkpoint_2 = torch.tensor([0, 1, 2, 3, 4])
            torch.save(checkpoint_1, self.model_checkpoints_path + "checkpoint_1.ckpt")
            torch.save(checkpoint_2, self.model_checkpoints_path + "checkpoint_2.ckpt")


Next, define a component that needs the checkpoints:

.. code:: python

    class ModelDeploy(lit.LightningWork):
        def __init__(self, ckpt_path, *args, **kwargs):
            super().__init__()
            self.ckpt_path = ckpt_path

        def run(self):
            ckpts = os.list_dir(self.ckpt_path)
            checkpoint_1 = torch.load(ckpts[0])
            checkpoint_2 = torch.load(ckpts[1])

Link both components via a parent component:

.. code:: python

    class Root(lit.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train = ModelTraining()
            self.deploy = ModelDeploy(ckpt_path=self.train.model_checkpoints_path)

        def run(self):
            self.train.run()
            self.deploy.run()


    app = lit.LightningApp(Root())


For example, here we save a file on one component and use it in another component:

.. code:: python

    from lightning_app.storage.path import Path


    class ComponentA(LightningWork):
        def __init__(self):
            super().__init__()
            self.boring_path = Path("boring_file.txt")

        def run(self):
            # This should be used as a REFERENCE to the file.
            self.boring_path = Path("boring_file.txt")
            with open(self.boring_path, "w") as f:
                f.write(FILE_CONTENT)
