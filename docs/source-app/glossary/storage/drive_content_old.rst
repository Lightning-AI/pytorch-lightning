:orphan:


************
About Drives
************

Lightning Drive storage makes it easy to share files between LightningWorks so you can run your Lightning App both locally and in the cloud without changing the code.

The Drive object provides a central place for your components to share data.

The Drive acts as an isolate folder and any component can access it by knowing its name.

Your components can put, list, get, and delete files from and to the Drive (except LightningFlows).

----

***********************
What Drive does for you
***********************

Think of every instance of the Drive object acting like a Google Drive or like Dropbox.

By sharing the Drive between components through the LightningFlow,
several components can have a shared place to read and write files from.

----

**************
Create a Drive
**************

In order to create a Drive, you simply need to pass its name with the prefix ``lit://`` as follows:

.. code-block:: python

    from lightning.app.storage import Drive

    # The identifier of this Drive is ``drive_1``
    # Note: You need to add Lightning protocol ``lit://`` as a prefix.

    drive_1 = Drive("lit://drive_1")

    # The identifier of this Drive is ``drive_2``
    drive_2 = Drive("lit://drive_2")

Any components can create a drive object.

.. code-block:: python

    from lightning.app import LightningFlow, LightningWork
    from lightning.app.storage import Drive


    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.drive_1 = Drive("lit://drive_1")

        def run(self):
            ...


    class Work(LightningWork):
        def __init__(self):
            super().__init__()
            self.drive_1 = Drive("lit://drive_1")

        def run(self):
            ...

----

*****************************
Supported actions with Drives
*****************************

A Drive supports put, list, get, and delete actions.

.. code-block:: python

    from lightning.app.storage import Drive

    drive = Drive("lit://drive")

    drive.list(".")  # Returns [] as empty

    # Created file.
    with open("a.txt", "w") as f:
        f.write("Hello World !")

    drive.put("a.txt")

    drive.list(".")  # Returns ["a.txt"] as the file copied in the Drive during the put action.

    drive.get("a.txt")  # Get the file into the current worker

    drive.delete("a.txt")

    drive.list(".")  # Returns [] as empty

----

**********************************
Component interactions with Drives
**********************************

Here is an illustrated code example on how to create drives within works.

.. figure::  https://pl-public-data.s3.amazonaws.com/assets_lightning/drive_2.png

.. code-block:: python

    from lightning.app import LightningFlow, LightningWork, LightningApp
    from lightning.app.storage import Drive


    class Work_A(LightningWork):
        def __init__(self):
            super().__init__()
            # The identifier of the Drive is ``drive_1``
            # Note: You need to add Lightning protocol ``lit://`` as a prefix.
            self.drive_1 = Drive("lit://drive_1")

        def run(self):
            # 1. Create a file.
            with open("a.txt", "w") as f:
                f.write("Hello World !")

            # 2. Put the file into the drive.
            self.drive_1.put("a.txt")


    class Work_B(LightningWork):
        def __init__(self):
            super().__init__()

            # Note: Work B has access 2 drives.

            # The identifier of this Drive is ``drive_1``
            self.drive_1 = Drive("lit://drive_1")
            # The identifier of this Drive is ``drive_2``
            self.drive_2 = Drive("lit://drive_2")

        def run(self):
            # 1. Create a file.
            with open("b.txt", "w") as f:
                f.write("Hello World !")

            # 2. Put the file into both drives.
            self.drive_1.put("b.txt")
            self.drive_2.put("b.txt")


    class Work_C(LightningWork):
        def __init__(self):
            super().__init__()
            self.drive_2 = Drive("lit://drive_2")

        def run(self):
            # 1. Create a file.
            with open("c.txt", "w") as f:
                f.write("Hello World !")

            # 2. Put the file into the drive.
            self.drive_2.put("c.txt")

----

*****************************
Transfer files with Drive
*****************************

In the example below, the Drive is created by the flow and passed to its LightningWork's.

The ``Work_1`` put a file **a.txt** in the **Drive("lit://this_drive_id")** and the ``Work_2`` can list and get the **a.txt** file from it.

.. literalinclude:: ../../../../examples/app/drive/app.py


----

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Learn about the Path Object.
   :description: Transfer Files From One Component to Another by Reference.
   :col_css: col-md-4
   :button_link: path.html
   :height: 180
   :tag: Intermediate

.. raw:: html

        </div>
    </div>
