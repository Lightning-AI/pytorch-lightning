:orphan:

##########################
Other Cluster Environments
##########################

**Audience**: Users who want to run on a cluster that launches the training script via MPI, LSF, Kubeflow, etc.

Lightning automates the details behind training on the most common cluster environments.
While :doc:`SLURM <./slurm>` is the most popular choice for on-prem clusters, there are other systems that Lightning can detect automatically.

Don't have access to an enterprise cluster? Try the :doc:`Lightning cloud <./cloud>`.


----


***
MPI
***

`MPI (Message Passing Interface) <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ is a communication system for parallel computing.
There are many implementations available, the most popular among them are `OpenMPI <https://www.open-mpi.org/>`_ and `MPICH <https://www.mpich.org/>`_.
To support all these, Lightning relies on the `mpi4py package <https://github.com/mpi4py/mpi4py>`_:

.. code-block:: bash

    pip install mpi4py

If the package is installed and the Python script gets launched by MPI, Fabric will automatically detect it and parse the process information from the environment.
There is nothing you have to change in your code:

.. code-block:: python

    fabric = Fabric(...)  # automatically detects MPI
    print(fabric.world_size)  # world size provided by MPI
    print(fabric.global_rank)  # rank provided by MPI
    ...

If you want to bypass the automatic detection, you can explicitly set the MPI environment as a plugin:

.. code-block:: python

    from lightning.fabric.plugins.environments import MPIEnvironment

    fabric = Fabric(..., plugins=[MPIEnvironment()])


----


***
LSF
***

Coming soon.


----


********
Kubeflow
********

Coming soon.
