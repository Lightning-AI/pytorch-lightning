## Distributed, Low-Precision Transformer Example

This example shows how to use `ModelParallelStrategy` in `Fabric` to train a Transformer model minimizing memory usage, maximizing throughput, and distributing load across multiple GPUs.

### Training Large Models and Memory Requirements

One of the main challenges when training large models, like large language models (LLMs), is dealing with their memory footprint. LLMs can be so large that weights, activations, gradients and optimizer state don't fit a single GPU, so that they need to be distributed across multiple GPUs, and across multiple machines. There are multiple ways of distributing computations, among which fully-sharded data parallelism (FSDP) and tensor parallelism (TP).

An additional way of reducing memory requirements is representing floating point numbers in weights and activations in low numerical precision, such as 16-bit (`bfloat16`), or 8-bit (`fp8`). This leads to savings in memory usage, as well as memory bandwidth usage (fewer bytes transferred from device memory to GPU cores in unit time).

Roughly, reducing precision to `fp8` for linear layers can lead to 2x reduction in memory requirements and 1.6x improvement in throughput. Support for `fp8` weights and activations requires recent GPUs - Hopper, Ada Lovelace and above (e.g. H100, L4, L40).

The introduction of tensor subclasses in PyTorch brought two new APIs that can be used to achieve memory savings and distributed training (as well as inference) in combination:

- [torch ao](https://github.com/pytorch/ao) to execute linear layers in low numerical precision (`fp8` and other quantized formats)
- [dtensors](https://pytorch.org/docs/stable/distributed.tensor.html) to distribute models across GPUs, by combining TP and FSDP (referred to FSDP2 in PyTorch)

Notably, `torch ao` introduces quantization and dequantization operations in the model that may result in slow-downs if not optimized. Using `torch.compile` after `torch ao` recovers performance by generating optimized kernels for those operations.

### Vanilla Transformer Example

This example shows how to train a vanilla Transformer model using `fp8` precision and the FSDP2 distributed strategy, and then optimize the resulting model through `torch.compile`.

Specifically, we employ the `ModelParallelStrategy`, which accepts a `parallelize_fn` to distribute the model using the PyTorch DTensor API.
We use the same function to also pass the model through the `torch ao` API (prior to FSDP2), as well as `torch.compile` (after FSDP2).

The resulting code follows the PyTorch API closely, while also taking advantage of the rest of Lightning Fabric.

To execute the code directly just run:

```bash
python train.py
```

### A Note on torch.compile

Note that Fabric also supports calling `torch.compile` on a model and passing it to `fabric.setup_model` or `fabric.setup_model_and_optimizers`.

While this works well, in order to get the most out of the combination of the latest distributed, quantization, and compile PyTorch API's, we recommend invoking `torch.compile` as part of the `parallelize_fn` argument of `ModelParallelStrategy`, as shown in this example.
