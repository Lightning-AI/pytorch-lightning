############################################
Deploy models into production (intermediate)
############################################
**Audience**: Researchers and MLEs looking to use their models for predictions without Lightning dependencies.

----

*********************
Use PyTorch as normal
*********************
If you prefer to use PyTorch directly, feel free to use any Lightning checkpoint without Lightning.

.. code-block:: python

    import torch


    class MyModel(nn.Module):
        ...


    model = MyModel()
    checkpoint = torch.load("path/to/lightning/checkpoint.ckpt")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

----

********************************************
Extract nn.Module from Lightning checkpoints
********************************************
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

    model_weights = checkpoint["state_dict"]

    # update keys by dropping `auto_encoder.`
    for key in list(model_weights):
        model_weights[key.replace("auto_encoder.", "")] = model_weights.pop(key)

    model.load_state_dict(model_weights)
    model.eval()
    x = torch.randn(1, 64)

    with torch.no_grad():
        y_hat = model(x)
