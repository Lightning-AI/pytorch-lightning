## Tensor Parallel and 2D Parallel

This example shows how to apply tensor-parallelism to your model (here Llama 3 7B) with the `ModelParallelStrategy`, and how it can be combined with FSDP (2D parallelism).
PyTorch 2.3+ and a machine with at least 4 GPUs and 24 GB memory each are required to run this example.

```bash
pip install 'torch>=2.3'
```

Navigate to this example folder and run the training script:

```bash
cd examples/fabric/tensor_parallel
python train.py
```

You should see an output like this:

```
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 4 processes
----------------------------------------------------------------------------------------------------

Number of model parameters: 6.7 B
Starting training ...
Iteration 0 complete
Iteration 1 complete
Iteration 2 complete
Iteration 3 complete
Iteration 4 complete
Iteration 5 complete
Iteration 6 complete
Iteration 7 complete
Saving a (distributed) checkpoint ...
Training successfully completed!
Peak memory usage: 17.95 GB
```

> \[!NOTE\]
> The `ModelParallelStrategy` is experimental and subject to change. Report issues on [GitHub](https://github.com/Lightning-AI/pytorch-lightning/issues).
