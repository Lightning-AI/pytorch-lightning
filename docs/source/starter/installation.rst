.. _installation:

############
Installation
############

PyTorch Lightning is maintained and tested on Pytorch 3.7.0+ and PyTorch 1.7.0+.

*********************
Installation with Pip
*********************

Install any supported version of PyTorch if you want from `PyTorch Installation Page <https://pytorch.org/get-started/locally/#start-locally>`_.
Now you can install using `pip <https://pypi.org/project/pytorch-lightning/>`_ using the following command:

.. code-block:: bash

    pip install pytorch-lightning


***********************
Installation with Conda
***********************

If you don't have conda installed, follow the `Conda Installation Guide <https://docs.conda.io/projects/conda/en/latest/user-guide/install>`_.
Lightning can be installed with `conda <https://anaconda.org/conda-forge/pytorch-lightning>`_ using the following command:

.. code-block:: bash

    conda install pytorch-lightning -c conda-forge

You can also use `Conda Environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_:

.. code-block:: bash

    conda activate my_env
    conda install pytorch-lightning -c conda-forge

************************
Installation from Source
************************

Install future patch release from the source. Note that patch release contains only the bugfixes for the recent major release.

.. code-block:: bash

    pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@release/1.5.x --upgrade


Install nightly from the source. Note that nightly contains all the bugfixes and newly released features that
are not published yet. Newly added features might contain bugs so no gurantess.

.. code-block:: bash

    pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip
