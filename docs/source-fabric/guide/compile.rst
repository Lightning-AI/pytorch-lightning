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

    You should compile the model **before** calling ``fabric.setup()`` as shown above for an optimal integration with features in Fabric.

The newly added call to ``torch.compile()`` by itself doesn't do much yet. It just wraps the model in a "compiled model".
The actual optimization will start when calling ``forward()`` on the model for the first time:

.. code-block:: python

    input = torch.randint(0, 128, (4, 256), device=fabric.device)
    target = torch.randint(0, 128, (4, 256), device=fabric.device)

    output = model(input, target)  # compiles when `forward()` runs for the first time


----


******************
Avoid graph breaks
******************

When ``torch.compile`` looks at the code in your model's ``forward()`` method, it will try to compile as much of the code as possible.
If there are regions in the code that it doesn't understand, it will introduce a so-called "graph break" that essentially splits the code in optimized and unoptimized parts.
Graph breaks aren't a deal breaker, since the optimized parts will still run faster.
But if you want to get the most out of ``torch.compile``, you might want to invest rewriting the problematic section of the code that produce the breaks.

You can check whether your model produces graph breaks by calling ``torch.compile`` with ``fullraph=True``:

.. code-block:: python

    # Force an error if there is a graph break in the model
    model = torch.compile(model, fullgraph=True)

The error messages produced here are often quite cryptic.

----


***********************************
Experiment with compilation options
***********************************



----


*********************************************************
(Experimental) Apply `torch.compile` over FSDP, DDP, etc.
*********************************************************

.. code-block:: python

    # Choose a distributed strategy like DDP or FSDP
    fabric = L.Fabric(devices=2, strategy="ddp")

    # Compile the model
    model = torch.compile(model)

    # Default: `fabric.setup()` will not reapply the compilation over DDP/FSDP
    model = fabric.setup(model, _reapply_compile=False)

    # Recompile the model over DDP/FSDP (experimental)
    model = fabric.setup(model, _reapply_compile=True)
