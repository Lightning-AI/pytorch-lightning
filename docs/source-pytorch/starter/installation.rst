:orphan:

.. _installation:

############
Installation
############



*****************************
Apple Silicon (M1/M2/M3) Macs
*****************************
While ML related python packages are updated to work with Apple Silicon, you'll need to set 2 environment variables on install.

.. code-block:: bash

    # needed for M1/M2/M3
    export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
    export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

    python -m pip install -U lightning


----

****************
Install with pip
****************

Install lightning inside a virtual env or conda environment with pip

.. code-block:: bash

    python -m pip install lightning

--------------

******************
Install with Conda
******************

If you don't have conda installed, follow the `Conda Installation Guide <https://docs.conda.io/projects/conda/en/latest/user-guide/install>`_.
Lightning can be installed with `conda <https://anaconda.org/conda-forge/pytorch-lightning>`_ using the following command:

.. code-block:: bash

    conda install lightning -c conda-forge

You can also use `Conda Environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_:

.. code-block:: bash

    conda activate my_env
    conda install lightning -c conda-forge

----

*****************
Build from Source
*****************

Install nightly from the source. Note that it contains all the bug fixes and newly released features that
are not published yet. This is the bleeding edge, so use it at your own discretion.

.. code-block:: bash

    pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip -U

Install future patch releases from the source. Note that the patch release contains only the bug fixes for the recent major release.

.. code-block:: bash

    pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/release/stable.zip -U

----

*******************************
Optimized for model development
*******************************
If you are deploying models built with Lightning in production and require few dependencies, try using the optimized `lightning[pytorch]` package:

.. code-block:: bash

    pip install lightning

^^^^^^^^^^^^^^^^^^^^^^
Custom PyTorch Version
^^^^^^^^^^^^^^^^^^^^^^
To use any PyTorch version visit the `PyTorch Installation Page <https://pytorch.org/get-started/locally/#start-locally>`_.

----


*******************************************
Optimized for ML workflows (lightning Apps)
*******************************************
If you are deploying workflows built with Lightning in production and require fewer dependencies, try using the optimized `lightning[apps]` package:

.. code-block:: bash

    pip install lightning-app
