:orphan:

.. _installation:

############
Installation
############

****************
Install with pip
****************

Install lightning inside a virtual env or conda environment with pip

.. code-block:: bash

    python -m pip install lightning


----


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


In case you face difficulty with pulling the GRPC package, please follow this `thread <https://stackoverflow.com/questions/66640705/how-can-i-install-grpcio-on-an-apple-m1-silicon-laptop>`_


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



^^^^^^^^^^^^^^^^^^^^^^
Custom PyTorch Version
^^^^^^^^^^^^^^^^^^^^^^

To use any PyTorch version visit the `PyTorch Installation Page <https://pytorch.org/get-started/locally/#start-locally>`_.
You can find the list of supported PyTorch versions in our :ref:`compatibility matrix <versioning:Compatibility matrix>`.

----


*******************************************
Optimized for ML workflows (Lightning Apps)
*******************************************
If you are deploying workflows built with Lightning in production and require fewer dependencies, try using the optimized ``lightning[apps]`` package:

.. code-block:: bash

    pip install lightning-app
