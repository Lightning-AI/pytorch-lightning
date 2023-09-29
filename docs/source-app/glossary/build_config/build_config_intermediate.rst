:orphan:

##################################
Build Configuration (Intermediate)
##################################

**Audience:** Users who need to execute commands to configure the machine before a LightningWork runs on it.

**Level:** Intermediate

When a LightningWork machine starts up in the cloud, it uses a lightweight operating system with essential packages pre-installed.
If you need to install additional system packages or run other configuration steps before your code executes on that machine, it is possible to do so by creating a custom
:class:`~lightning.app.utilities.packaging.build_config.BuildConfig`:

1.  Subclass :class:`~lightning.app.utilities.packaging.build_config.BuildConfig`:

    .. code-block:: python

        from lightning.app import BuildConfig


        @dataclass
        class CustomBuildConfig(BuildConfig):
            def build_commands(self):
                return ["sudo apt-get install libsparsehash-dev"]


2.  Set the build config on the LightningWork:

    .. code-block:: python

        from lightning.app import LightningWork


        class MyWork(LightningWork):
            def __init__(self):
                super().__init__()

                # Use the custom build config
                self.cloud_build_config = CustomBuildConfig()

                # Can also be combined with extra requirements
                self.cloud_build_config = CustomBuildConfig(requirements=["torchmetrics"])

.. note::
    - When you need to execute commands or install tools that require more privileges than the current user has, you can use ``sudo`` without needing to provide a password, e.g., when installing system packages.
    - The build config only applies when running in the cloud and gets ignored otherwise. A local build config is currently not supported.

.. warning::
     Custom base images are not supported with the default CPU cloud compute. For example:

     .. code-block:: py

         class MyWork(LightningWork):
             def __init__(self):
              super().__init__(cloud_build_config=BuildConfig(image="my-custom-image")) # no cloud compute, for example default work
