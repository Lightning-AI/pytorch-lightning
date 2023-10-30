#######
Compile
#######

Compiling your PyTorch model can result in significant speedups, especially on the latest hardware such as NVIDIA GPUs.
This guide shows you how to apply `torch.compile` correctly in your code.

.. note::
    This requires PyTorch >= 2.0.


----


***********************************
Apply `torch.compile` to your model
***********************************

Compiling a model in a script together with Fabric is as simple as adding one line of code, calling :func:`torch.compile`:

.. code-block:: python

    import torch
    import lightning as L
    from lightning.pytorch.demos import Transformer

    fabric = L.Fabric(devices=1)
    model = Transformer(128)

    # Compile the model
    model = torch.compile(model)

    # `fabric.setup()` should come after `torch.compile()`
    model = fabric.setup(model)


.. important::

    Since :func:`torch.compile` does not yet reliably support DDP, FSDP and the like, we recommend compiling the model **before** calling ``fabric.setup()``.

The newly added call to ``torch.compile()`` by itself doesn't do much yet. It just wraps the model in a "compiled model".
The actual optimization will start when calling ``forward()`` on the model for the first time:

.. code-block:: python

    input = torch.randint(0, 128, (4, 256), device=fabric.device)
    target = torch.randint(0, 128, (4, 256), device=fabric.device)

    output = model(input, target)  # compiles when `forward()` runs for the first time


