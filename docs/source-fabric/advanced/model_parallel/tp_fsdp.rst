##########################################
2D Parallelism (Tensor Parallelism + FSDP)
##########################################

2D Parallelism combines Tensor Parallelism (TP) and Fully Sharded Data Parallelism (FSDP) to leverage the memory efficiency of FSDP and the computational scalability of TP.
This hybrid approach balances the trade-offs of each method, optimizing memory usage and minimizing communication overhead, enabling the training of extremely large models on large GPU clusters.

The :doc:`Tensor Parallelism documentation <tp>` and a general understanding of `FSDP <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`_ are a prerequisite for this tutorial.

.. raw:: html

    <a target="_blank" href="https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-lightning-fabric">
      <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio" style="width: auto; max-width: none;"/>
    </a>


----


*********************
Enable 2D parallelism
*********************

We will start off with the same feed forward example model as in the :doc:`Tensor Parallelism tutorial <tp>`.

.. code-block:: python

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
More on how the device mesh works later.

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

By writing the parallelization code in a separate function rather than hardcoding it into the model, we keep the original source code clean and maintainable.
In addition to the tensor-parallel code from the :doc:`Tensor Parallelism tutorial <tp>`, this function also shards the model's parameters using FSDP along the data-parallel dimension.

Finally, pass the parallelization function to the :class:`~lightning.fabric.strategies.model_parallel.ModelParallelStrategy` and configure the data-parallel and tensor-parallel sizes:

.. code-block:: python

    import lightning as L
    from lightning.fabric.strategies import ModelParallelStrategy

    strategy = ModelParallelStrategy(
        parallelize_fn=parallelize_feedforward,
        # Define the size of the 2D parallelism
        # Set these to "auto" (default) to apply TP intra-node and FSDP inter-node
        data_parallel_size=2,
        tensor_parallel_size=2,
    )

    fabric = L.Fabric(accelerator="cuda", devices=4, strategy=strategy)
    fabric.launch()


In this example with 4 GPUs, Fabric will create a device mesh that groups GPU 0-1 and GPU 2-3 (2 groups because ``data_parallel_size=2``, and 2 GPUs per group because ``tensor_parallel_size=2``).
Later on when ``fabric.setup(model)`` is called, each layer wrapped with FSDP (``fully_shard``) will be split into two shards, one for the GPU 0-1 group, and one for the GPU 2-3 group.
Finally, the tensor parallelism will apply to each group, splitting the sharded tensor across the GPUs within each group.


.. collapse:: Full training example (requires at least 4 GPUs).

    .. code-block:: python

        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
        from torch.distributed.tensor.parallel import parallelize_module
        from torch.distributed._composable.fsdp.fully_shard import fully_shard

        import lightning as L
        from lightning.pytorch.demos.boring_classes import RandomDataset
        from lightning.fabric.strategies import ModelParallelStrategy


        class FeedForward(nn.Module):
            def __init__(self, dim, hidden_dim):
                super().__init__()
                self.w1 = nn.Linear(dim, hidden_dim, bias=False)
                self.w2 = nn.Linear(hidden_dim, dim, bias=False)
                self.w3 = nn.Linear(dim, hidden_dim, bias=False)

            def forward(self, x):
                return self.w2(F.silu(self.w1(x)) * self.w3(x))


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


        strategy = ModelParallelStrategy(
            parallelize_fn=parallelize_feedforward,
            data_parallel_size=2,
            tensor_parallel_size=2,
        )

        fabric = L.Fabric(accelerator="cuda", devices=4, strategy=strategy)
        fabric.launch()

        # Initialize the model
        model = FeedForward(8192, 8192)
        model = fabric.setup(model)

        # Define the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        optimizer = fabric.setup_optimizers(optimizer)

        # Define dataset/dataloader
        dataset = RandomDataset(8192, 128)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        dataloader = fabric.setup_dataloaders(dataloader)

        # Simplified training loop
        for i, batch in enumerate(dataloader):
            output = model(batch)
            loss = output.sum()
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            fabric.print(f"Iteration {i} complete")

        fabric.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

.. note:: 2D Parallelism in Lightning Fabric as well as PyTorch is experimental. The APIs may change in the future.

Beyond this toy example, we recommend you study our `LLM 2D Parallel Example (Llama 3) <https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples/fabric/tensor_parallel>`_.


----


*******************
Effective use cases
*******************

In the toy example above, the parallelization is configured to work within a single machine across multiple GPUs.
However, in practice the main use case for 2D parallelism is in multi-node training, where one can effectively combine both methods to maximize throughput and model scale.
Since tensor-parallelism requires blocking collective calls, fast GPU data transfers are essential to keep throughput high and therefore TP is typically applied across GPUs within a machine.
On the other hand, FSDP by design has the advantage that it can overlap GPU transfers with the computation (it can prefetch layers).
Hence, combining FSDP for inter-node parallelism and TP for intra-node parallelism is generally a good strategy to minimize both the latency and network bandwidth usage, making it possible to scale to much larger models than is possible with FSDP alone.


.. code-block:: python

    from lightning.fabric.strategies import ModelParallelStrategy

    strategy = ModelParallelStrategy(
        # Default is "auto"
        # Applies TP intra-node and DP inter-node
        data_parallel_size="auto",
        tensor_parallel_size="auto",
    )


----


***************************
Data-loading considerations
***************************

In a tensor-parallelized model, it is important that the model receives an identical input on each GPU that participates in the same tensor-parallel group.
However, across the data-parallel dimension, the inputs should be different.
In other words, if TP is applied within a node, and FSDP across nodes, each node must receive a different batch, but every GPU within the node gets the same batch of data.

If you use a PyTorch data loader and set it up using :meth:`~lightning.fabric.fabric.Fabric.setup_dataloaders`, Fabric will automatically handle this for you by configuring the distributed sampler.
However, when you shuffle data in your dataset or data loader, or when applying randomized transformations/augmentations in your data, you must still ensure that the seed is set appropriately.


.. code-block:: python

    import lightning as L

    fabric = L.Fabric(...)

    # Define dataset/dataloader
    # If there is randomness/augmentation in the dataset, fix the seed
    dataset = MyDataset(seed=42)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Fabric configures the sampler automatically for you such that
    # all batches in a tensor-parallel group are identical,
    # while still sharding the dataset across the data-parallel group
    dataloader = fabric.setup_dataloaders(dataloader)

    for i, batch in enumerate(dataloader):
        ...




----


**********
Next steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: LLM 2D Parallel Example
    :description: Full example how to combine TP + FSDP in a large language model (Llama 3)
    :col_css: col-md-4
    :button_link: https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples/fabric/tensor_parallel
    :height: 160
    :tag: advanced

.. displayitem::
    :header: Pipeline Parallelism
    :description: Coming sooon
    :col_css: col-md-4
    :height: 160
    :tag: advanced


.. raw:: html

        </div>
    </div>

|
