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
To write a file, first create a reference to the file with the :class:`~lightning.app.storage.path.Path` class, then write to it:

.. code:: python

    from lightning.app.storage import Path

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
    ******************************
    TODO

    ----

*********************************
Example: Share a model checkpoint
*********************************
A common workflow in ML is to use a checkpoint created by another component.
First, define a component that saves a checkpoint:

.. literalinclude:: ./share_files_between_components/app.py
    :lines: -19

Next, define a component that needs the checkpoints:

.. literalinclude:: ./share_files_between_components/app.py
    :lines: 20-31

Link both components via a parent component:

.. literalinclude:: ./share_files_between_components/app.py
    :lines: 32-


Run the app above with the following command:

.. code-block:: bash

    lightning run app docs/source/workflows/share_files_between_components/app.py

.. code-block:: console

    Your Lightning App is starting. This won't take long.
    INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
    Loaded checkpoint_1: tensor([0, 1, 2, 3, 4])
    Loaded checkpoint_2: tensor([0, 1, 2, 3, 4])


For example, here we save a file on one component and use it in another component:

.. code:: python

    from lightning.app.storage import Path


    class ComponentA(LightningWork):
        def __init__(self):
            super().__init__()
            self.boring_path = None

        def run(self):
            # This should be used as a REFERENCE to the file.
            self.boring_path = Path("boring_file.txt")
            with open(self.boring_path, "w") as f:
                f.write(FILE_CONTENT)
