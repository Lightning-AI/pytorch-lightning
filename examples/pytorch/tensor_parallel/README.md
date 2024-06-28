## Tensor Parallel and 2D Parallel

This example shows how to apply tensor-parallelism to your model (here Llama 3 7B) with the `ModelParallelStrategy`, and how it can be combined with FSDP (2D parallelism).
PyTorch 2.3+ and a machine with at least 4 GPUs and 24 GB memory each are required to run this example.

```bash
pip install 'torch>=2.3'
```

Navigate to this example folder and run the training script:

```bash
cd examples/pytorch/tensor_parallel
python train.py
```

You should see an output like this:

```
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs

Number of model parameters: 6.7 B
Starting training ...

Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 4 processes
----------------------------------------------------------------------------------------------------

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3]

Epoch 0: 100%|█████████████████████████████████████████████| 10/10 [01:49<00:00, 0.09it/s, v_num=2]
`Trainer.fit` stopped: `max_epochs=1` reached.                                      
Saving a (distributed) checkpoint ...
Training successfully completed!
Peak memory usage: 36.73 GB
```

> \[!NOTE\]
> The `ModelParallelStrategy` is experimental and subject to change. Report issues on [GitHub](https://github.com/Lightning-AI/pytorch-lightning/issues).
