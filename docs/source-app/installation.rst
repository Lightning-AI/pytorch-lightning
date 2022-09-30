
.. _install:


############
Installation
############
**Prerequisites**: Use Python 3.8.x or later (3.8.x, 3.9.x, 3.10.x). We also recommend you install in a virtual environment (`learn how <install_beginner.rst>`_).

----

*****************************
Apple Silicon (M1/M2/M3) Macs
*****************************
Until ML related python packages are updated to work with Apple Silicon, you'll need to set 2 environment variables on install.

.. code-block:: bash

    # needed for M1/M2/M3
    export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
    export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

    python -m pip install -U lightning

----

****************
Install with pip
****************

.. code:: bash

    python -m pip install -U lightning

If you encounter issues during installation use the following to help troubleshoot:

.. code:: bash

        pip list | grep lightning

----

******************
Install on Windows
******************
To install on Windows, make sure you:

- have Pip
- Git
- setup an alias for Python: python=python3
- Add the root folder of Lightning to the Environment Variables to PATH