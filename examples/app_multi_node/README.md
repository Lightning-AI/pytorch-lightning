# Lightning & Multi Node Training

Lightning supports makes multi-node training simple by providing a simple interface to orchestrate compute and data.

## Multi Node with raw PyTorch

You can run the multi-node raw PyTorch by running the following commands.

```bash
lightning run app app_torch_work.py
```

Here are the logs you obtain:

```
Global Rank 0: [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
Global Rank 1: [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
Global Rank 2: [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
Global Rank 3: [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
```

## Multi Node with PyTorch Lightning

Lightning supports running PyTorch Lightning from a script or within a Lightning Work.

### Multi Node PyTorch Lightning Script

```bash
lightning run app app_pl_script.py
```

### Multi Node PyTorch Lightning Work

```bash
lightning run app app_pl_work.py
```

## Multi Node with any frameworks

```bash
lightning run app app_generic_work.py
```
