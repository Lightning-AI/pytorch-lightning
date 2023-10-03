:orphan:

**************************
What are Lightning Drives?
**************************

Lightning Drives are shared app storage that allow you to share files between :doc:`LightningWork (Work) <../../core_api/lightning_work/index>` components, so that you distributed components can share files when running on the cloud. Using drives, you can run your Lightning App both locally and in the cloud without changing the code.

The Drive object provides a central place for your components to share data.

The Drive acts as an isolated folder and any component can access it by knowing its name.

We currently support two types of Drives: Lightning-managed (``lit://``) and S3 (``s3://``).

+-----------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| Lightning-managed (``lit://``)    | Allows read-write operations and are accessible through the Drive API from a Work.                                            |
|                                   |                                                                                                                               |
|                                   | They allow your components to put, list, get, and delete files from and to the Drive (except LightningFlows).                 |
+-----------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| S3 (``s3://``)                    | S3 is AWS S3 storage mounted at a filesystem mount point. S3 is read-only (for now) and its primary purpose is                |
|                                   | to give you a permanent location to access your training data.                                                                |
|                                   |                                                                                                                               |
|                                   | They allow your components to list and get files located on the Drive.                                                        |
+-----------------------------------+-------------------------------------------------------------------------------------------------------------------------------+

----

**********************
What Drives do for you
**********************

Think of every instance of the Drive object acting like a Google Drive or like Dropbox.

By sharing the Drive between components through the LightningFlow,
several components can have a shared place to read (S3 Drives) or read and write (Lightning-managed Drives) files from.

S3 Drive Limitations
^^^^^^^^^^^^^^^^^^^^

These limitations only apply to S3 Drives:

* There is no top level “shareable” S3 drive object. Each S3 Drive is owned by a particular Work.  However, it’s possible to create a Drive with the same location across multiple Works.

* S3 buckets cannot be mounted as Drives once a Work has been instantiated. The `Drive` object must be initialized passed to a Work at creation time.

* Whenever a Drive is mounted to a Work, an indexing process will be done again for the provided S3 bucket. This may lead to performance issues with particularly large S3 buckets. For context, 1M files with 2-3 levels of nesting takes less than 1 second to index.

----

**************
Create a Drive
**************

In order to create a Drive, you simply need to pass its name with the prefix ``lit://`` or ``s3://``.

.. note:: We do not support mounting single objects for S3 buckets, so there must be a trailing `/` in the s3:// URL. For example: ``s3://foo/bar/``.

.. code-block:: python

    from lightning.app.storage import Drive

    # The identifier of this Drive is ``drive_1``
    # Note: You need to add Lightning protocol ``lit://`` as a prefix.

    drive_1 = Drive("lit://drive_1")

    # The identifier of this Drive is ``drive_2``
    drive_2 = Drive("s3://drive_2/")

Any component can create a drive object for ``lit://`` Drives.

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

A Lightning-managed Drive supports put, list, get, and delete actions.

An S3 Drive supports list and get actions (for now).

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

Here is an illustrated code example on how to create drives within Works.

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

*************************
Transfer files with Drive
*************************

In the example below, the Drive is created by the Flow and passed to its Works.

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
