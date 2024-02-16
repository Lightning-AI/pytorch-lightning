:orphan:

########################################
Deploy models into production (advanced)
########################################
**Audience**: Machine learning engineers optimizing models for enterprise-scale production environments.

----

*********************************
Compile your model to TorchScript
*********************************
`TorchScript <https://pytorch.org/docs/stable/jit.html>`_ allows you to serialize your models in a way that it can be loaded in non-Python environments.
The ``LightningModule`` has a handy method :meth:`~lightning.pytorch.core.LightningModule.to_torchscript` that returns a scripted module which you
can save or directly use.

.. testcode:: python

    class SimpleModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(in_features=64, out_features=4)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))


    # create the model
    model = SimpleModel()
    script = model.to_torchscript()

    # save for use in production environment
    torch.jit.save(script, "model.pt")

It is recommended that you install the latest supported version of PyTorch to use this feature without limitations.

Once you have the exported model, you can run it in PyTorch or C++ runtime:

.. code-block:: python

    inp = torch.rand(1, 64)
    scripted_module = torch.jit.load("model.pt")
    output = scripted_module(inp)


If you want to script a different method, you can decorate the method with :func:`torch.jit.export`:

.. code-block:: python

    class LitMCdropoutModel(L.LightningModule):
        def __init__(self, model, mc_iteration):
            super().__init__()
            self.model = model
            self.dropout = nn.Dropout()
            self.mc_iteration = mc_iteration

        @torch.jit.export
        def predict_step(self, batch, batch_idx):
            # enable Monte Carlo Dropout
            self.dropout.train()

            # take average of `self.mc_iteration` iterations
            pred = [self.dropout(self.model(x)).unsqueeze(0) for _ in range(self.mc_iteration)]
            pred = torch.vstack(pred).mean(dim=0)
            return pred


    model = LitMCdropoutModel(...)
    script = model.to_torchscript(file_path="model.pt", method="script")
