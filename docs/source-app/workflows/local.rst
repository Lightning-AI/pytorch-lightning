##########
Run localy
##########

**Audience:** Users who want to run Lightning App on a their own machine.

----

***********
Run locally
***********

You can run lightning apps on any machine, using `lightning run app` CLI command.

.. code:: bash

    # start the app
    lightning run app app.py


.. note::

    When running a Lighting App on your local machine, any :class:`~lightning_app.utilities.packaging.cloud_compute.CloudCompute`
    configuration (including a :class:`~lightning_app.storage.mount.Mount`) is ignored at runtime. If you need access to
    these files on your local disk, you should download a copy of them to your machine.

----

*****************************
Run on remote machine via ssh
*****************************

To run a Lightning App on any machine, simply ssh to the machine and run the app directly.

.. code:: bash

    # Copy over credentials from your local machine to your cloud machine
    scp ~/.lightning/credentials.json your_name@your_cloud_machine:~/.lightning

    # log into your cloud machine
    ssh your_name@your_cloud_machine

    # get your code on the machine and install deps
    ...

    # start the app
    lightning run app app.py
