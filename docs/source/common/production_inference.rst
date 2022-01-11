.. _production_inference:

#######################
Inference in Production
#######################

Since :class:`~pytorch_lightning.core.lightning.LightningModule` is just :class:`torch.nn.Module`, all the possible inferencing techniques for deploying PyTorch models to production
apply here too. Lightning provides some helper methods to help you out with it.

---------------

**********
Using ONNX
**********

Lightning provides a handy function to quickly export your model to `ONNX <https://pytorch.org/docs/stable/onnx.html>`_ format
which allows the model to be independent of PyTorch and run on an ONNX Runtime.

To export your model to ONNX format call the :meth:`~pytorch_lightning.core.lightning.LightningModule.to_onnx` function on your :class:`~pytorch_lightning.core.lightning.LightningModule` with the ``filepath`` and ``input_sample``.

.. code-block:: python

    class SimpleModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(in_features=64, out_features=4)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))


    filepath = "model.onnx"
    model = SimpleModel()
    input_sample = torch.randn((1, 64))
    model.to_onnx(filepath, input_sample, export_params=True)

You can also skip passing the input sample if the ``example_input_array`` property is specified in your :class:`~pytorch_lightning.core.lightning.LightningModule`.

Once you have the exported model, you can run it on your ONNX runtime in the following way:

.. code-block:: python

    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: np.random.randn(1, 64)}
    ort_outs = ort_session.run(None, ort_inputs)

---------------

*****************
Using TorchScript
*****************

`TorchScript <https://pytorch.org/docs/stable/jit.html>`_ allows you to serialize your models in a way that it can be loaded in non-Python environments.
The ``LightningModule`` has a handy method :meth:`~pytorch_lightning.core.lightning.LightningModule.to_torchscript` that returns a scripted module which you
can save or directly use.

.. code-block:: python

    model = SimpleModel()
    script = model.to_torchscript()

    # save for use in production environment
    torch.jit.save(script, "model.pt")

It is recommended that you install the latest supported version of PyTorch to use this feature without limitations.

Once you have the exported model, you can run it in Pytorch or C++ runtime:

.. code-block:: python

    inp = torch.rand(1, 64)
    scripted_module = torch.jit.load("model.pt")
    output = scripted_module(dummy_input)

---------------

********************
Using Python Runtime
********************

You can also load the saved checkpoint inside a Python runtime and use it as a regular :class:`torch.nn.Module`.

.. code-block:: python

    model = SimpleModel()

    # train it
    trainer = Trainer(gpus=2)
    trainer.fit(model, train_dataloader, val_dataloader)

    # use model after training or load weights and drop into the production system
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
