:orphan:

###########################################
Communication between distributed processes
###########################################

Fabric enables you to easily access information about a process or send data between processes with a standardized API and agnostic to the distributed strategy.


----


*******************
Rank and world size
*******************

.. figure:: https://pl-flash-data.s3.amazonaws.com/fabric/docs/collectives/ranks.jpeg
   :alt: The different type of process ranks: Local, global, node.
   :width: 100%

.. code-block:: python

    # Devices and num_nodes determine how many processes there are
    fabric = Fabric(devices=2, num_nodes=3)

    # The total number of processes running across all devices and nodes
    fabric.world_size  # 2 * 3 = 6

    # The global index of the current process across all devices and nodes
    fabric.global_rank

    # The index of the current process among the processes running on the local node
    fabric.local_rank

    # The index of the current node
    fabric.node_rank

    # Whether this global rank is rank zero.
    if fabric.is_global_zero:
        # do something on rank 0
        ...



----


*********
Broadcast
*********

.. figure:: https://pl-flash-data.s3.amazonaws.com/fabric/docs/collectives/broadcast.jpeg
   :alt: The broadcast collective operation
   :width: 100%

.. code-block:: python

    fabric = Fabric(...)

    # Transfer an object from one process to all the others
    fabric.broadcast(..., src=...)



----


******
Gather
******

.. figure:: https://pl-flash-data.s3.amazonaws.com/fabric/docs/collectives/all-gather.jpeg
   :alt: The All-gather collective operation
   :width: 100%

.. code-block:: python

    fabric = Fabric(...)

    # Transfer and concatenate tensors across processes
    fabric.all_gather(...)


----


******
Reduce
******

.. figure:: https://pl-flash-data.s3.amazonaws.com/fabric/docs/collectives/all-reduce.jpeg
   :alt: The All-reduce collective operation
   :width: 100%

.. code-block:: python

    fabric = Fabric(...)

    # TODO
    fabric.all_reduce(...)


----


*******
Barrier
*******

.. figure:: https://pl-flash-data.s3.amazonaws.com/fabric/docs/collectives/barrier.jpeg
   :alt: The barrier for process synchronization
   :width: 100%

.. code-block:: python

    fabric = Fabric(accelerator="cpu", devices=4)
    fabric.launch()

    # Simulate each process taking a different amount of time
    sleep(2 * fabric.global_rank)

    # Wait for all processes to reach the barrier
    fabric.barrier()
    print("All processes synchronized!")
