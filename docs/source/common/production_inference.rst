.. _production_inference:

#######################
Inference in Production
#######################

Once a model is trained, deploying to production and running inference is the next task. To help you with it, here are the possible approaches
you can use to deploy and make inferences with your models.

------------

******************
With Lightning API
******************

The following are some possible ways you can use Lightning to run inference in production. Note that PyTorch Lightning has some extra dependencies and using raw PyTorch might be advantageous.
in your production environment.

------------

Prediction API
==============

Lightning provides you with a prediction API that can be accessed using :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`.
To configure this with your LightningModule, you would need to override the :meth:`~pytorch_lightning.core.lightning.LightningModule.predict_step` method.
By default :meth:`~pytorch_lightning.core.lightning.LightningModule.predict_step` calls the :meth:`~pytorch_lightning.core.lightning.LightningModule.forward`
method. In order to customize this behaviour, simply override the :meth:`~pytorch_lightning.core.lightning.LightningModule.predict_step` method. This can be useful to add some pre-processing or post-processing logic to your data.

For the example let's override ``predict_step`` and try out `Monte Carlo Dropout <https://arxiv.org/pdf/1506.02142.pdf>`_:

.. code-block:: python

    class LitMCdropoutModel(pl.LightningModule):
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

------------

PyTorch Runtime
===============

You can also load the saved checkpoint and use it as a regular :class:`torch.nn.Module`.

.. code-block:: python

    class SimpleModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(in_features=64, out_features=4)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))


    # create the model
    model = SimpleModel()

    # train it
    trainer = Trainer(accelerator="gpu", devices=2)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint("best_model.ckpt", weights_only=True)

    # use model after training or load weights and drop into the production system
    model = SimpleModel.load_from_checkpoint("best_model.ckpt")
    model.eval()
    x = torch.randn(1, 64)

    with torch.no_grad():
        y_hat = model(x)

------------

*********************
Without Lightning API
*********************

As the :class:`~pytorch_lightning.core.lightning.LightningModule` is simply a :class:`torch.nn.Module`, common techniques to export PyTorch models
to production apply here too. However, the :class:`~pytorch_lightning.core.lightning.LightningModule` provides helper methods to help you out with it.

------------

Convert to ONNX
===============

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


    # create the model
    model = SimpleModel()
    filepath = "model.onnx"
    input_sample = torch.randn((1, 64))
    model.to_onnx(filepath, input_sample, export_params=True)

You can also skip passing the input sample if the ``example_input_array`` property is specified in your :class:`~pytorch_lightning.core.lightning.LightningModule`.

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

------------

Convert to TorchScript
======================

`TorchScript <https://pytorch.org/docs/stable/jit.html>`_ allows you to serialize your models in a way that it can be loaded in non-Python environments.
The ``LightningModule`` has a handy method :meth:`~pytorch_lightning.core.lightning.LightningModule.to_torchscript` that returns a scripted module which you
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

Once you have the exported model, you can run it in Pytorch or C++ runtime:

.. code-block:: python

    inp = torch.rand(1, 64)
    scripted_module = torch.jit.load("model.pt")
    output = scripted_module(inp)


If you want to script a different method, you can decorate the method with :func:`torch.jit.export`:

.. code-block:: python

    class LitMCdropoutModel(pl.LightningModule):
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


------------

PyTorch Runtime
===============

You can also load the saved checkpoint and use it as a regular :class:`torch.nn.Module`. You can extract all your :class:`torch.nn.Module`
and load the weights using the checkpoint saved using LightningModule after training. For this, we recommend copying the exact implementation
from your LightningModule ``init`` and ``forward`` method.

.. code-block:: python

    class Encoder(nn.Module):
        ...


    class Decoder(nn.Module):
        ...


    class AutoEncoderProd(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def forward(self, x):
            return self.encoder(x)


    class AutoEncoderSystem(LightningModule):
        def __init__(self):
            super().__init__()
            self.auto_encoder = AutoEncoderProd()

        def forward(self, x):
            return self.auto_encoder.encoder(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.auto_encoder.encoder(x)
            y_hat = self.auto_encoder.decoder(y_hat)
            loss = ...
            return loss


    # train it
    trainer = Trainer(devices=2, accelerator="gpu", strategy="ddp")
    model = AutoEncoderSystem()
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint("best_model.ckpt")


    # create the PyTorch model and load the checkpoint weights
    model = AutoEncoderProd()
    checkpoint = torch.load("best_model.ckpt")
    hyper_parameters = checkpoint["hyper_parameters"]

    # if you want to restore any hyperparameters, you can pass them too
    model = AutoEncoderProd(**hyper_parameters)

    state_dict = checkpoint["state_dict"]

    # update keys by dropping `auto_encoder.`
    for key in list(model_weights):
        model_weights[key.replace("auto_encoder.", "")] = model_weights.pop(key)

    model.load_state_dict(model_weights)
    model.eval()
    x = torch.randn(1, 64)

    with torch.no_grad():
        y_hat = model(x)
