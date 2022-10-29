:orphan:

##################################
Installation on Apple Silicon Macs
##################################

Apple Silicon (M1, M2, M3) Mac environments need a bit of tweaking before you install.

----

****************
Install with pip
****************

Install the ``lightning`` package

    .. code:: bash

        export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
        export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

        python -m pip install -U lightning
