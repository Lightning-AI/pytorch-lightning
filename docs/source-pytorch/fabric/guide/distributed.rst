:orphan:

##################################
Working with distributed processes
##################################

Page is under construction.

----------


You can also easily use distributed collectives if required.

.. code-block:: python

    fabric = Fabric()

    # Transfer and concatenate tensors across processes
    fabric.all_gather(...)

    # Transfer an object from one process to all the others
    fabric.broadcast(..., src=...)

    # The total number of processes running across all devices and nodes.
    fabric.world_size

    # The global index of the current process across all devices and nodes.
    fabric.global_rank

    # The index of the current process among the processes running on the local node.
    fabric.local_rank

    # The index of the current node.
    fabric.node_rank

    # Whether this global rank is rank zero.
    if fabric.is_global_zero:
        # do something on rank 0
        ...

    # Wait for all processes to enter this call.
    fabric.barrier()


The code stays agnostic, whether you are running on CPU, on two GPUS or on multiple machines with many GPUs.

If you require custom data or model device placement, you can deactivate :class:`~lightning_fabric.fabric.Fabric`'s automatic placement by doing ``fabric.setup_dataloaders(..., move_to_device=False)`` for the data and ``fabric.setup(..., move_to_device=False)`` for the model.
Furthermore, you can access the current device from ``fabric.device`` or rely on :meth:`~lightning_fabric.fabric.Fabric.to_device` utility to move an object to the current device.
