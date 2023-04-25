:orphan:

##################################
Run on an on-prem cluster (expert)
##################################

.. _custom-cluster:

----

**************************
Integrate your own cluster
**************************

Lightning provides an interface for providing your own definition of a cluster environment. It mainly consists of
parsing the right environment variables to access information such as world size, global and local rank (process id),
and node rank (node id). Here is an example of a custom
:class:`~lightning.pytorch.plugins.environments.cluster_environment.ClusterEnvironment`:

.. code-block:: python

    import os
    from lightning.pytorch.plugins.environments import ClusterEnvironment


    class MyClusterEnvironment(ClusterEnvironment):
        @property
        def creates_processes_externally(self) -> bool:
            """Return True if the cluster is managed (you don't launch processes yourself)"""
            return True

        def world_size(self) -> int:
            return int(os.environ["WORLD_SIZE"])

        def global_rank(self) -> int:
            return int(os.environ["RANK"])

        def local_rank(self) -> int:
            return int(os.environ["LOCAL_RANK"])

        def node_rank(self) -> int:
            return int(os.environ["NODE_RANK"])

        def main_address(self) -> str:
            return os.environ["MASTER_ADDRESS"]

        def main_port(self) -> int:
            return int(os.environ["MASTER_PORT"])


    trainer = Trainer(plugins=[MyClusterEnvironment()])
