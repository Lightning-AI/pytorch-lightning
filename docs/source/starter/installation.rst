.. _installation:

############
Installation
############

--------------

******************
Lightning Coverage
******************

PyTorch Lightning is maintained and tested on different Python and PyTorch versions.

Check out the `CI Coverage <https://github.com/PyTorchLightning/pytorch-lightning#continuous-integration>`_ for more info.

It is rigorously tested across multiple GPUs, TPUs, CPUs and IPUs. GPU tests run on two NVIDIA P100. TPU tests run on Google GKE TPUv2/3.
TPU py3.7 means we support Colab and Kaggle env. IPU tests run on MK1 IPU boxes.

--------------

*********************
Installation with Pip
*********************

Install any supported version of PyTorch if you want from `PyTorch Installation Page <https://pytorch.org/get-started/locally/#start-locally>`_.
Now you can install using `pip <https://pypi.org/project/pytorch-lightning/>`_ using the following command:

.. code-block:: bash

    pip install pytorch-lightning

--------------

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

--------------

************************
Installation from Source
************************

Install nightly from the source. Note that it contains all the bug fixes and newly released features that
are not published yet. This is the bleeding edge, so use it at your own discretion.

.. code-block:: bash

    pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip

Install future patch releases from the source. Note that the patch release contains only the bug fixes for the recent major release.

.. code-block:: bash

    pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/refs/heads/release/1.5.x.zip
