:orphan:

#################################
Validate and test a model (basic)
#################################
**Audience**: Users who want to add a validation loop to avoid overfitting

----

***************
Add a test loop
***************
To make sure a model can generalize to an unseen dataset (ie: to publish a paper or in a production environment) a dataset is normally split into two parts, the *train* split and the *test* split.

The test set is **NOT** used during training, it is **ONLY** used once the model has been trained to see how the model will do in the real-world.

----

Find the train and test splits
==============================
Datasets come with two splits. Refer to the dataset documentation to find the *train* and *test* splits.

.. code-block:: python

   import torch.utils.data as data
   from torchvision import datasets
   import torchvision.transforms as transforms

   # Load data sets
   transform = transforms.ToTensor()
   train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
   test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)

----

Define the test loop
====================
To add a test loop, implement the **test_step** method of the LightningModule

.. code:: python

    class LitAutoEncoder(L.LightningModule):
        def training_step(self, batch, batch_idx):
            ...

        def test_step(self, batch, batch_idx):
            # this is the test loop
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            test_loss = F.mse_loss(x_hat, x)
            self.log("test_loss", test_loss)

----

Train with the test loop
========================
Once the model has finished training, call **.test**

.. code-block:: python

   from torch.utils.data import DataLoader

   # initialize the Trainer
   trainer = Trainer()

   # test the model
   trainer.test(model, dataloaders=DataLoader(test_set))

----

*********************
Add a validation loop
*********************
During training, it's common practice to use a small portion of the train split to determine when the model has finished training.

----

Split the training data
=======================
As a rule of thumb, we use 20% of the training set as the **validation set**. This number varies from dataset to dataset.

.. code-block:: python

   # use 20% of training data for validation
   train_set_size = int(len(train_set) * 0.8)
   valid_set_size = len(train_set) - train_set_size

   # split the train set into two
   seed = torch.Generator().manual_seed(42)
   train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

----

Define the validation loop
==========================
To add a validation loop, implement the **validation_step** method of the LightningModule

.. code:: python

    class LitAutoEncoder(L.LightningModule):
        def training_step(self, batch, batch_idx):
            ...

        def validation_step(self, batch, batch_idx):
            # this is the validation loop
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            val_loss = F.mse_loss(x_hat, x)
            self.log("val_loss", val_loss)

----

Train with the validation loop
==============================
To run the validation loop, pass in the validation set to **.fit**

.. code-block:: python

   from torch.utils.data import DataLoader

   train_loader = DataLoader(train_set)
   valid_loader = DataLoader(valid_set)
   model = LitAutoEncoder(...)

   # train with both splits
   trainer = L.Trainer()
   trainer.fit(model, train_loader, valid_loader)
