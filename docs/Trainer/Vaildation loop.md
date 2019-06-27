The lightning validation loop handles everything except the actual computations of your model. To decide what will happen in your validation loop, define the [validation_step function](../../Pytorch-lightning/LightningModule/#validation_step).

Below are all the things lightning automates for you in the validation loop.