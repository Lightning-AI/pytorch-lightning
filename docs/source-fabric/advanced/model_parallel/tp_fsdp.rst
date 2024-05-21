##########################################
2D Parallelism (Tensor Parallelism + FSDP)
##########################################


The :doc:`Tensor Parallelism documentation <tp>` is a prerequisite for this tutorial.

We will start off with the same toy example model as in the :doc:`Tensor Parallelism tutorial <tp>`.

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class FeedForward(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

Next, we define a function that applies the desired parallelism to our model.
The function must take as first argument the model and as second argument the a :class:`~torch.distributed.device_mesh.DeviceMesh`.

.. code-block:: python

    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    from torch.distributed.tensor.parallel import parallelize_module
    from torch.distributed._composable.fsdp.fully_shard import fully_shard

    def parallelize_feedforward(model, device_mesh):
        # Lightning will set up a device mesh for you
        # Here, it is 2-dimensional
        tp_mesh = device_mesh["tensor_parallel"]
        dp_mesh = device_mesh["data_parallel"]

        if tp_mesh.size() > 1:
            # Use PyTorch's distributed tensor APIs to parallelize the model
            plan = {
                "w1": ColwiseParallel(),
                "w2": RowwiseParallel(),
                "w3": ColwiseParallel(),
            }
            parallelize_module(model, tp_mesh, plan)

        if dp_mesh.size() > 1:
            # Use PyTorch's FSDP2 APIs to parallelize the model
            fully_shard(model.w1, mesh=dp_mesh)
            fully_shard(model.w2, mesh=dp_mesh)
            fully_shard(model.w3, mesh=dp_mesh)
            fully_shard(model, mesh=dp_mesh)

        return model


Finally, pass the parallelization function to the strategy and configure the data-parallel and tensor-parallel sizes:

.. code-block:: python

    import lightning as L
    from lightning.fabric.strategies import ModelParallelStrategy

    strategy = ModelParallelStrategy(
        parallelize_fn=parallelize_feedforward,
        # Define the size of the 2D parallelism
        # Set to "auto" to apply TP intra-node and DP inter-node
        data_parallel_size=2,
        tensor_parallel_size=2,
    )

    fabric = L.Fabric(accelerator="cuda", devices=4, strategy=strategy)
    fabric.launch()


----


***************
The device mesh
***************
