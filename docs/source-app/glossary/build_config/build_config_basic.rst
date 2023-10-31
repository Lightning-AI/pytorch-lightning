:orphan:

###########################
Build Configuration (Basic)
###########################

**Audience:** Users who need to install Python packages for an individual LightningWork.

**Level:** Basic

----

***********************************
List dependencies in separate files
***********************************

If you are building an app with multiple LightningWorks that have different or even conflicting requirements, split your dependencies into individual files
for more granular control.

.. code-block:: bash

    ├── app.py
    ├── requirements.txt          # Global requirements for the entire app
    └── works
        ├── serve
        │   ├── requirements.txt  # Requirements specific to the 'serve' work
        │   └── serve.py          # Source file for the LightningWork
        └── train
            ├── requirements.txt  # Requirements specific to the 'train' work
            └── train.py          # Source file for the LightningWork

The requirements.txt file must be located in the same directory as the source file of the LightningWork.
When the LightningWork starts up, it will pick up the requirements file if present and install all listed packages.

.. note::
    This only applies when running in the cloud. The requirements.txt files get ignored when running locally.

----

***********************************
Define the requirements in the code
***********************************

Instead of listing the requirements in a file, you can also pass them to the LightningWork at runtime using the
:class:`~lightning.app.utilities.packaging.build_config.BuildConfig`:

.. code-block:: python
    :emphasize-lines: 7

    from lightning.app import LightningWork, BuildConfig


    class MyWork(LightningWork):
        def __init__(self):
            super().__init__()
            self.cloud_build_config = BuildConfig(requirements=["torch>=1.8", "torchmetrics"])

.. note::
    The build config only applies when running in the cloud and gets ignored otherwise. A local build config is currently not supported.

.. warning::
     Custom base images are not supported with the default CPU cloud compute. For example:

     .. code-block:: py

         class MyWork(LightningWork):
             def __init__(self):
              super().__init__(cloud_build_config=BuildConfig(image="my-custom-image")) # no cloud compute, for example default work
