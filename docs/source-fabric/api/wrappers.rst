#########################
Modules wrapped by Fabric
#########################

When you :doc:`set up <../api/fabric_methods>` a model in Fabric, it gets automatically wrapped by a new module, the ``FabricModule``:

.. code-block:: python

    import torch
    import lightning as L

    fabric = L.Fabric()
    model = torch.nn.Linear(10, 2)
    model = fabric.setup(model)

    print(type(model))  # <class 'lightning.fabric.wrappers._FabricModule'>

This wrapper module takes care of a few things for you, notably:

- Strategy: Handles strategy-specific logic for the forward method (DDP, FSDP, etc.).
- Precision: Inputs and outputs passed through ``forward`` get automatically converted to the right precision depending on the ``Fabric(precision=...)`` setting.
- Device: The wrapper remembers which device the model is on, you can access `model.device`.

The ``FabricModule`` wrapper is completely transparent and most users will never need to interact with it directly.
Below we describe a few functions and properties of the wrapper for advanced use cases.
This might be useful if you are building a custom Trainer using Fabric as the core.


----


********************************
Accessing methods and attributes
********************************

Access to methods and attributes gets redirected to the original model automatically:

.. code-block:: python

    import torch
    import lightning as L

    fabric = L.Fabric()
    model = torch.nn.Linear(10, 2)
    fabric_model = fabric.setup(model)

    # You can access attributes and methods normally
    print(fabric_model.weight is model.weight)  # True


----


********************
Unwrapping the model
********************

You can check whether a model is wrapped in a ``FabricModule`` with the ``is_wrapped`` utility function:

.. code-block:: python

    import torch
    import lightning as L
    from lightning.fabric import is_wrapped

    fabric = L.Fabric()
    model = torch.nn.Linear(10, 2)
    fabric_model = fabric.setup(model)

    print(is_wrapped(model))  # False
    print(is_wrapped(fabric_model))  # True


If you ever need to, you can access the original model explicitly via ``.module``:

.. code-block:: python

    # Access the original model explicitly
    original_model = fabric_model.module

    print(original_model is model)  # True

