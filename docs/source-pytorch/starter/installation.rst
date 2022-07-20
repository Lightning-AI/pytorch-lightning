:orphan:

.. _installation:

############
Installation
############

--------------

****************
Install with pip
****************

Install any supported version of PyTorch if you want from `PyTorch Installation Page <https://pytorch.org/get-started/locally/#start-locally>`_.
Now you can install using `pip <https://pypi.org/project/pytorch-lightning/>`_ using the following command:

.. code-block:: bash

    pip install pytorch-lightning

--------------

******************
Install with Conda
******************

If you don't have conda installed, follow the `Conda Installation Guide <https://docs.conda.io/projects/conda/en/latest/user-guide/install>`_.
Lightning can be installed with `conda <https://anaconda.org/conda-forge/pytorch-lightning>`_ using the following command:

.. code-block:: bash

    conda install pytorch-lightning -c conda-forge

You can also use `Conda Environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_:

.. code-block:: bash

    conda activate my_env
    conda install pytorch-lightning -c conda-forge

--------------

*****************
Build from Source
*****************

Install nightly from the source. Note that it contains all the bug fixes and newly released features that
are not published yet. This is the bleeding edge, so use it at your own discretion.

.. code-block:: bash

    pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip

Install future patch releases from the source. Note that the patch release contains only the bug fixes for the recent major release.

.. code-block:: bash

    pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/refs/heads/release/1.5.x.zip
