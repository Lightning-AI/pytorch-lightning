:orphan:

TPU training (Advanced)
=======================
**Audience:** Users looking to apply advanced performance techniques to TPU training.

----

Weight Sharing/Tying
--------------------
Weight Tying/Sharing is a technique where in the module weights are shared among two or more layers.
This is a common method to reduce memory consumption and is utilized in many State of the Art
architectures today.

PyTorch XLA requires these weights to be tied/shared after moving the model
to the TPU device. To support this requirement Lightning provides a model hook which is
called after the model is moved to the device. Any weights that require to be tied should
be done in the `on_post_move_to_device` model hook. This will ensure that the weights
among the modules are shared and not copied.

PyTorch Lightning has an inbuilt check which verifies that the model parameter lengths
match once the model is moved to the device. If the lengths do not match Lightning
throws a warning message.

Example:

.. code-block:: python

    from pytorch_lightning.core.module import LightningModule
    from torch import nn
    from pytorch_lightning.trainer.trainer import Trainer


    class WeightSharingModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(32, 10, bias=False)
            self.layer_2 = nn.Linear(10, 32, bias=False)
            self.layer_3 = nn.Linear(32, 10, bias=False)
            # TPU shared weights are copied independently
            # on the XLA device and this line won't have any effect.
            # However, it works fine for CPU and GPU.
            self.layer_3.weight = self.layer_1.weight

        def forward(self, x):
            x = self.layer_1(x)
            x = self.layer_2(x)
            x = self.layer_3(x)
            return x

        def on_post_move_to_device(self):
            # Weights shared after the model has been moved to TPU Device
            self.layer_3.weight = self.layer_1.weight


    model = WeightSharingModule()
    trainer = Trainer(max_epochs=1, accelerator="tpu", devices=8)

See `XLA Documentation <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#xla-tensor-quirks>`_

----

XLA
---
XLA is the library that interfaces PyTorch with the TPUs.
For more information check out `XLA <https://github.com/pytorch/xla>`_.

Guide for `troubleshooting XLA <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md>`_
