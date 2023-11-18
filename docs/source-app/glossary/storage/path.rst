:orphan:

############
Path Storage
############

**Audience:** Users who want to share files between components.


The Lightning Storage system makes it easy to share files between LightningWork so you can run your app both locally and in the cloud without changing the code.

----

***********************
What is a Path Object ?
***********************

The Path object is a reference to a specific file or directory from a LightningWork and can be used to transfer those files to another LightningWork (one way, from source to destination).

A good mental representation of the Path Object usage is the `relay race <https://en.wikipedia.org/wiki/Relay_race>`_.
To make a transfer, the receiver asks (e.g when the path object is passed by the flow to the receiver)
for a copy of the files (baton) owned by their producer (e.g the LightningWork which created the files).

.. figure::  https://pl-public-data.s3.amazonaws.com/assets_lightning/path2.png

*******************************************
How does the Path Object works internally ?
*******************************************

To understand the Path Object internal, let's first answer this question: How do you locate a specific file or folder within a distributed system made of multiple machines ?

You need to know on which machine the file or folder is located (e.g the LightningWork name uniquely identify its own machine in the cloud) and
then you need the local path of the file or folder on that machine.

In simple words, the Lightning Path augments :class:`pathlib.Path` object by tracking on which machine the file or folder is located.

----

**************************
When to use Path storage ?
**************************

In the cloud, every :class:`~lightning.app.core.work.LightningWork` runs in a separate machine with its own filesystem.
This means files in one Work cannot be directly accessed in another like you would be able to when running the app locally.
But with Lightning Storage, this is easy: Simply declare which files need to be shared and Lightning will take care of the rest.

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/path.mp4
    :width: 600
    :autoplay:
    :loop:
    :muted:


----


***********************************
Tell Lightning where your files are
***********************************

Convert every filesystem path you want to share with other LightningWorks to by adding ``lit://`` in front of it.

.. code-block:: python

    from lightning.app import LightningWork
    from lightning.app.storage import Path


    class SourceWork(LightningWork):
        def __init__(self):
            super().__init__()
            self.checkpoint_dir = None

        def run(self):
            # Normally you would do:
            # self.checkpoint_dir = "outputs/checkpoints"
            # os.makedirs("outputs/checkpoints")
            # ...

            # In Lightning, do:
            self.checkpoint_dir = "lit://outputs/checkpoints"
            os.makedirs(self.checkpoint_dir)
            ...


Under the hood, we convert this string to a :class:`~lightning.app.storage.path.Path` object, which is a drop-in replacement for :class:`pathlib.Path` meaning it will work with :mod:`os`, :mod:`os.path` and :mod:`pathlib` filesystem operations out of the box!


----


****************************
Access files in another Work
****************************

Accessing files from another LightningWork is as easy as handing the path over by reference.
For example, share a directory by passing it as an input to the run method of the destination work:

.. code-block:: python
    :emphasize-lines: 12

    from lightning.app import LightningFlow


    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.source = SourceWork()
            self.destination = DestinationWork()

        def run(self):
            self.source.run()
            # Pass the Path reference from one work to another
            self.destination.run(self.source.checkpoint_dir)


When the destination Work starts, Lightning will automatically transfer the files to its filesystem (if they exist on the other end):

.. code-block:: python

    class DestinationWork(LightningWork):
        def run(self, checkpoint_dir):
            # The directory is now accessible inside this Work
            files = os.listdir(checkpoint_dir)
            ...


The automatic transfer only happens if the referenced files already exist in the originating LightningWork and it will overwrite any files that already exist locally.
In all other cases, you can trigger the transfer manually.


----


******************
Get files manually
******************

If you need to access files at a specific time or transfer them multiple times, use ``.get()`` method:

.. code-block:: python

        def run(self, checkpoint_dir):
            ...
            # Make the directory available
            checkpoint_dir.get()

            # If the path already exists locally, you can force overwriting it
            checkpoint_dir.get(overwrite=True)

            files = os.listdir(checkpoint_dir)
            ...


Multiple calls to the ``.get()`` method will always result in file transfers, regardless of whether the files have changed or not.
If the path does not exist remotely, it will raise a ``FileNotFoundError``.
If you need to handle this case, the Path also offers a method to check if files exist remotely.

----


********************************
Check if a file or folder exists
********************************

You can check if a path exists locally or remotely in the source Work using the ``.exists_local()`` and ``.exists_remote()`` methods:

.. code-block:: python

        def run(self, checkpoint_dir):
            if checkpoint_dir.exists_remote():
                # Get the file only if it exists in the source Work
                checkpoint_dir.get()

            # OR

            if checkpoint_dir.exists_local():
                # Do something with the file if it exists locally
                files = os.listdir(checkpoint_dir)


----


*************
Persist files
*************

If a LightningWork finishes or stops due to an interruption (e.g., due to insufficient credits), the filesystem and all files in it get deleted (unless running locally).
Lightning makes sure all Paths that are part of the state get stored and made accessible to the other Works that still need these files.

.. code-block:: python

    from lightning.app.storage import Path


    class Work(LightningWork):
        def __init__(self):
            super().__init__()
            # The files in this path will be saved as an artifact when the Work finishes
            self.checkpoint_dir = "lit://outputs/checkpoints"

            # The files in this path WON'T be saved because it is not declared as a Lightning Path
            self.log_dir = "outputs/logs"


----


*********************************
Example: Share a model checkpoint
*********************************

A common workflow in ML is to use a checkpoint created by another component.
First, define a component that saves a checkpoint:

.. code:: python
    :emphasize-lines: 14-18

    from lightning.app import LightningFlow, LightningWork
    from lightning.app.storage import Path
    import torch
    import os


    class ModelTraining(LightningWork):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.checkpoint_dir = "lit://outputs/checkpoints"

        def run(self):
            # create a directory
            os.makedirs(self.model_checkpoints_path, exist_ok=True)
            # make fake checkpoints
            checkpoint_1 = torch.tensor([0, 1, 2, 3, 4])
            checkpoint_2 = torch.tensor([0, 1, 2, 3, 4])
            torch.save(checkpoint_1, os.path.join(self.checkpoint_dir, "checkpoint_1.ckpt"))
            torch.save(checkpoint_2, os.path.join(self.checkpoint_dir, "checkpoint_2.ckpt"))


Next, define a component that needs the checkpoints:

.. code:: python
    :emphasize-lines: 4, 7

    class ModelDeploy(LightningWork):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def run(self, checkpoint_dir):
            ckpts = os.listdir(checkpoint_dir)
            checkpoint_1 = torch.load(ckpts[0])
            checkpoint_2 = torch.load(ckpts[1])

Link both components via a parent component:

.. code:: python
    :emphasize-lines: 7

    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.train = ModelTraining()

            # pass the checkpoint path
            self.deploy = ModelDeploy()

        def run(self):
            self.train.run()
            self.deploy.run(checkpoint_dir=self.train.checkpoint_dir)


    app = L.LightningApp(Flow())


----

**************************
Frequently Asked Questions
**************************

- **Q: Can files in a LightningWork be accessed inside the LightningFlow too?**

  No, LightningFlow is intentionally designed not to perform filesystem operations and computations and is intended to exclusively orchestrate Flow and Work.

- **Q: Is it possible to reference any file using the Lightning lit:// path notation?**

  Yes, but only files for which the app has write permissions can be copied from Work to Work (apps don't run with root privileges).

- **Q: Can I access the Lightning Storage in my UI (StreamLit, Web, ...)?**

  This is currently not supported but will be in the future.

- **Q: Should I define my lit:// path in the __init__ or the run method?**

  You can declare a Lightning path anywhere you'd like. However, the ``.get()`` and ``.exists_*()`` methods only work inside of the run method of a LightningWork.

- **Q:How often does Lightning synchronize the files between my Work?**

  Lightning does not synchronize the files between works. It only transfers the files once when the Work ``run`` method starts.
  But you can call ``Path.get()`` as many times as you wish to transfer the latest file into the current Work.

- **Does Lightning provide me direct access to the shared cloud folder?**

  No, and this is on purpose. This restriction forces developers to build modular components that can be shared and integrated
  into apps easily. This would be much harder to achieve if file paths in these components would reference a global shared storage.

----

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Learn about the Drive Object.
   :description: Put, List and Get Files From a Shared Drive Disk.
   :col_css: col-md-4
   :button_link: drive.html
   :height: 180
   :tag: Basic

.. raw:: html

        </div>
    </div>
