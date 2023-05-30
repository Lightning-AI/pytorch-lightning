:orphan:

##############################
Build Configuration (Advanced)
##############################

**Audience:** Users who want full control over the docker image that is being installed in the cloud.

**Level:** Advanced

Advanced users who need full control over the environment a LightningWork runs in can specify a custom docker image that will be deployed in the cloud.


----

******************
Use a docker image
******************

Create a :class:`~lightning.app.utilities.packaging.build_config.BuildConfig` and provide a **publicly accessible** link to where the image is hosted:

.. code-block:: python

    from lightning.app import LightningWork, BuildConfig


    class MyWork(LightningWork):
        def __init__(self):
            super().__init__()

            # Using a publicly hosted docker image:
            self.cloud_build_config = BuildConfig(
                # This is one of the base images Lightning uses by default
                image="ghcr.io/gridai/base-images:v1.8-gpu"
            )

            # Can also be combined with extra requirements
            self.cloud_build_config = BuildConfig(image="...", requirements=["torchmetrics"])


.. warning::
    Many public hosters like DockerHub apply rate limits for public images. We recommend to pull images from your own registry.
    For example, you can set up a
    `docker registry on GitHub <https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry>`_.


.. note::
    - The build config only applies when running in the cloud and gets ignored otherwise. A local build config is currently not supported.
    - Images from private registries are currently not supported.

.. note::
    Custom docker images must have python installed. We'll use `virtualenv` from this system python to create a virtual environment.
    We'll also configure the `virtualenv` to use the packages installed under system's python so your packages are not lost

----


*********************
Provide a docker file
*********************

.. note::
    Not yet supported. Coming soon.
