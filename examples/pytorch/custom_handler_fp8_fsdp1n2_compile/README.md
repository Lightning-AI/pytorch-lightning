# PyTorch Native FP8 Training with FSDP1/2 and Torch Compile using Custom Handler

This is an example of a ...

## Requirements

Install requirements by running

```bash
sh setup.sh
```

## Example

In this example we present

```bash
# # config the PYTHONPATH if needed
# export PYTHONPATH=/teamspace/studios/this_studio/pytorch-lightning/examples/pytorch/custom_handler_fp8_fsdp1n2_compile:$PYTHONPATH
cd pytorch-lightning/examples/pytorch/custom_handler_fp8_fsdp1n2_compile

# fsdp1 + fp8 + torch compile + gradient checkpointing + cpu offloading
python train.py  --enable_fp8 --enable_torch_compile --enable_gradient_checkpointing --enable_cpu_offload

# fsdp2 + fp8 + torch compile + gradient checkpointing (the example does not implement fsdp2 cpu offloading currently)
python train.py --enable_fsdp2 --enable_fp8 --enable_torch_compile --enable_gradient_checkpointing
```

## Test the handlers

```bash
# # config the PYTHONPATH if needed
# export PYTHONPATH=/teamspace/studios/this_studio/pytorch-lightning/examples/pytorch/custom_handler_fp8_fsdp1n2_compile:$PYTHONPATH
cd pytorch-lightning/examples/pytorch/custom_handler_fp8_fsdp1n2_compile
pytest tests/*
```

> **Warning**
