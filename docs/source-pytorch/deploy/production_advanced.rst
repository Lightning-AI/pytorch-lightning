########################################
Deploy models into production (advanced)
########################################
**Audience**: Machine learning engineers optimizing models for enterprise-scale production environments.

----

**************************
Compile your model to ONNX
**************************
`ONNX <https://pytorch.org/docs/stable/onnx.html>`_ is a package developed by Microsoft to optimize inference. ONNX allows the model to be independent of PyTorch and run on any ONNX Runtime.

To export your model to ONNX format call the :meth:`~lightning.pytorch.core.LightningModule.to_onnx` function on your :class:`~lightning.pytorch.core.LightningModule` with the ``filepath`` and ``input_sample``.

.. code-block:: python

    import torch
    from lightning.pytorch import LightningModule


    class SimpleModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(in_features=64, out_features=4)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

By default, the exported model uses the shape of the example input.
To serve different batch sizes, pass ``dynamo=True`` and ``dynamic_shapes`` to the exporter. This requires ``onnx`` and ``onnxscript``.
In this model, ``x`` is the name of the argument to ``forward``. Mark its first dimension as dynamic and keep the 64 features fixed.
Use a sample batch of at least two because PyTorch specializes dimensions of size one during export.

.. code-block:: python

    model = SimpleModel().eval()
    filepath = "model.onnx"
    input_sample = torch.randn(2, 64)
    dynamic_shapes = {"x": {0: torch.export.Dim("batch")}}
    model.to_onnx(filepath, input_sample, dynamo=True, dynamic_shapes=dynamic_shapes)

You can use the same exported model with different batch sizes in ONNX Runtime.

.. code-block:: python

    import onnxruntime

    session = onnxruntime.InferenceSession(filepath)
    input_name = session.get_inputs()[0].name
    for batch_size in (1, 8):
        x = torch.randn(batch_size, 64)
        (output,) = session.run(None, {input_name: x.numpy()})
        assert output.shape == (batch_size, 4)

To use the module's ``example_input_array`` for export, set it before calling ``to_onnx``.

.. code-block:: python

    model.example_input_array = input_sample
    model.to_onnx(filepath, dynamo=True, dynamic_shapes=dynamic_shapes)

----

****************************
Validate a Model Is Servable
****************************

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

Production ML Engineers would argue that a model shouldn't be trained if it can't be deployed reliably and in a fully automated manner.

In order to ease transition from training to production, PyTorch Lightning provides a way for you to validate a model can be served even before starting training.

In order to do so, your LightningModule needs to subclass the :class:`~lightning.pytorch.serve.servable_module.ServableModule`, implements its hooks and pass a :class:`~lightning.pytorch.serve.servable_module_validator.ServableModuleValidator` callback to the Trainer.

Below you can find an example of how the serving of a resnet18 can be validated.

.. literalinclude:: ../../../examples/pytorch/servable_module/production.py
