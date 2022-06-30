########################################
Deploy models into production (advanced)
########################################
**Audience**: Machine learning engineers optimizing models for enterprise-scale production environments.

----

**************************
Compile your model to ONNX
**************************
`ONNX <https://pytorch.org/docs/stable/onnx.html>`_ is a package developed by Microsoft to optimize inference. ONNX allows the model to be independent of PyTorch and run on any ONNX Runtime.

To export your model to ONNX format call the :meth:`~pytorch_lightning.core.module.LightningModule.to_onnx` function on your :class:`~pytorch_lightning.core.module.LightningModule` with the ``filepath`` and ``input_sample``.

.. code-block:: python

    class SimpleModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(in_features=64, out_features=4)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))


    # create the model
    model = SimpleModel()
    filepath = "model.onnx"
    input_sample = torch.randn((1, 64))
    model.to_onnx(filepath, input_sample, export_params=True)

You can also skip passing the input sample if the ``example_input_array`` property is specified in your :class:`~pytorch_lightning.core.module.LightningModule`.

.. code-block:: python

    class SimpleModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(in_features=64, out_features=4)
            self.example_input_array = torch.randn(7, 64)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))


    # create the model
    model = SimpleModel()
    filepath = "model.onnx"
    model.to_onnx(filepath, export_params=True)

Once you have the exported model, you can run it on your ONNX runtime in the following way:

.. code-block:: python

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: np.random.randn(1, 64)}
    ort_outs = ort_session.run(None, ort_inputs)
