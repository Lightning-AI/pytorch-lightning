# Lightning & Multi Node Training

Lightning supports makes multi-node training simple by providing a simple interface to orchestrate compute and data.

## Multi Node with raw PyTorch

You can run the multi-node raw PyTorch by running the following commands.

Here is an example where you spawn your processes yourself.

```bash
lightning run app train_pytorch.py
```

or you can use the built-in component for it.

```bash
lightning run app train_pytorch_spawn.py
```

## Multi Node with raw PyTorch + Fabric

You can run the multi-node raw PyTorch and Fabric by running the following commands.

```bash
lightning run app train_fabric.py
```

Using Fabric, you retain control over your loops while accessing in a minimal way all Lightning distributed strategies.

## Multi Node with Lightning Trainer

Lightning supports running Lightning Trainer from a script or within a Lightning Work.

You can either run a script directly

```bash
lightning run app train_pl_script.py
```

or run your code within as a work.

```bash
lightning run app train_pl.py
```

## Multi Node with any frameworks

```bash
lightning run app train_any.py
```
