:orphan:

########################################
Deploy models into production (advanced)
########################################
**Audience**: Machine learning engineers optimizing models for enterprise-scale production environments.

----

************************************
Export your model with torch.export
************************************

`torch.export <https://pytorch.org/docs/stable/export.html>`_ is the recommended way to capture PyTorch models for
deployment in production environments. It produces a clean intermediate representation with strong soundness guarantees,
making models suitable for inference optimization and cross-platform deployment.
You can export any ``LightningModule`` using the ``torch.export.export()`` API.

.. testcode:: python

    import torch
    from torch.export import export

    class SimpleModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(in_features=64, out_features=4)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))


    # create the model and example input
    model = SimpleModel()
    example_input = torch.randn(1, 64)

    # export the model
    exported_program = export(model, (example_input,))

    # save for use in production environment
    torch.export.save(exported_program, "model.pt2")

It is recommended that you install the latest supported version of PyTorch to use this feature without
limitations. Once you have the exported model, you can load and run it:

.. code-block:: python

    inp = torch.rand(1, 64)
    loaded_program = torch.export.load("model.pt2")
    output = loaded_program.module()(inp)


For more complex models, you can also export specific methods by creating a wrapper:

.. code-block:: python

    class LitMCdropoutModel(L.LightningModule):
        def __init__(self, model, mc_iteration):
            super().__init__()
            self.model = model
            self.dropout = nn.Dropout()
            self.mc_iteration = mc_iteration

        def predict_step(self, batch, batch_idx):
            # enable Monte Carlo Dropout
            self.dropout.train()

            # take average of `self.mc_iteration` iterations
            pred = [self.dropout(self.model(x)).unsqueeze(0) for _ in range(self.mc_iteration)]
            pred = torch.vstack(pred).mean(dim=0)
            return pred


    model = LitMCdropoutModel(...)
    example_batch = torch.randn(32, 10)  # example input

    # Export the predict_step method
    exported_program = torch.export.export(
        lambda batch, idx: model.predict_step(batch, idx),
        (example_batch, 0)
    )
    torch.export.save(exported_program, "mc_dropout_model.pt2")
