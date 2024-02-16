:orphan:

##############
Add Cloud Data
##############

**Audience:** Users who want to read files stored in a Cloud Object Bucket in an app.

******************************
Mounting Public AWS S3 Buckets
******************************

===================
Add Mount to a Work
===================

To mount data from a cloud bucket to your app compute, initialize a :class:`~lightning.app.storage.mount.Mount`
object with the source path of the s3 bucket and the absolute directory path where it should be mounted and
pass the :class:`~lightning.app.storage.mount.Mount` to the :class:`~lightning.app.utilities.packaging.cloud_compute.CloudCompute`
of the :class:`~lightning.app.core.work.LightningWork` it should be mounted on.

In this example, we will mount an S3 bucket: ``s3://ryft-public-sample-data/esRedditJson/`` to ``/content/esRedditJson/``.

.. code-block:: python

    from lightning.app import CloudCompute
    from lightning.app.storage import Mount

    self.my_work = MyWorkClass(
        cloud_compute=CloudCompute(
            mounts=Mount(
                source="s3://ryft-public-sample-data/esRedditJson/",
                mount_path="/content/esRedditJson/",
            ),
        )
    )

You can also pass multiple mounts to a single work by passing a ``List[Mount(...), ...]`` to the
``CloudCompute(mounts=...)`` argument.

.. note::

    * Mounts supported up to 1 Million files, 5GB per file. Need larger mounts? Contact support@lightning.ai
    * When adding multiple mounts, each one should have a unique ``mount_path``.
    * A maximum of 10 :class:`~lightning.app.storage.mount.Mount`\s can be added to a :class:`~lightning.app.core.work.LightningWork`.

=======================
Read Files From a Mount
=======================

Once a :class:`~lightning.app.storage.mount.Mount` object is passed to :class:`~lightning.app.utilities.packaging.cloud_compute.CloudCompute`,
you can access, list, or read any file from the mount under the specified ``mount_path``, just like you would if it
was on your local machine.

Assuming your ``mount_path`` is ``"/content/esRedditJson/"`` you can do the following:

----------
Read Files
----------

.. code-block:: python

    with open("/content/esRedditJson/esRedditJson1", "r") as f:
        some_data = f.read()

    # do something with "some_data"...

----------
List Files
----------

.. code-block:: python

    files = os.listdir("/content/esRedditJson/")

--------------------
See the Full Example
--------------------

.. code-block:: python
    :emphasize-lines: 10,15

    import os

    import lightning as L
    from lightning.app import CloudCompute
    from lightning.app.storage import Mount

    class ReadMount(L.LightningWork):
       def run(self):
           # Print a list of files stored in the mounted S3 Bucket.
           files = os.listdir("/content/esRedditJson/")
           for file in files:
               print(file)

           # Read the contents of a particular file in the bucket "esRedditJson1"
           with open("/content/esRedditJson/esRedditJson1", "r") as f:
               some_data = f.read()
               # do something with "some_data"...

    class Flow(L.LightningFlow):
       def __init__(self):
           super().__init__()
           self.my_work = ReadMount(
               cloud_compute=CloudCompute(
                   mounts=Mount(
                       source="s3://ryft-public-sample-data/esRedditJson/",
                       mount_path="/content/esRedditJson/",
                   ),
               )
           )

       def run(self):
           self.my_work.run()

.. note::

    When running a Lightning App on your local machine, any :class:`~lightning.app.utilities.packaging.cloud_compute.CloudCompute`
    configuration (including a :class:`~lightning.app.storage.mount.Mount`) is ignored at runtime. If you need access to
    these files on your local disk, you should download a copy of them to your machine.

.. note::

    Mounted files from an S3 bucket are ``read-only``. Any modifications, additions, or deletions
    to files in the mounted directory will not be reflected in the cloud object store.

----

**********************************************
Mounting Private AWS S3 Buckets - Coming Soon!
**********************************************

We'll Let you know when this feature is ready!

----

************************************************
Mounting Google Cloud GCS Buckets - Coming Soon!
************************************************

We'll Let you know when this feature is ready!
