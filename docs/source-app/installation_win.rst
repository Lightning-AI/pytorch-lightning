:orphan:

#######################
Installation on Windows
#######################

We strongly recommend to create a virtual environment first.
Don't know what this is? Follow our `beginner guide here <install_beginner.rst>`_.

Windows environments might need a little more tweaking before you install.

**Requirements**

* Python 3.8.x or later (3.8.x, 3.9.x, 3.10.x)
* Pip (the latest versions of Python will already have this)
* Git
* PyTorch - https://pytorch.org/get-started/locally/
* Setup an alias for Python: python=python3
* Add the root folder of Lightning to the Environment Variables to PATH
* Install Z shell (zsh) (This is required for Windows to install the quickstart app)

----

****************
Install with pip
****************

0.  Activate your virtual environment.

1.  Install the ``lightning`` package

    .. code:: bash

        python -m pip install -U lightning
