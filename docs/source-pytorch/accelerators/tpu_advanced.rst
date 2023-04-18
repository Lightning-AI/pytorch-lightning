:orphan:

TPU training (Advanced)
=======================
**Audience:** Users looking to apply advanced performance techniques to TPU training.

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

----

Weight Sharing/Tying
--------------------
Weight Tying/Sharing is a technique where in the module weights are shared among two or more layers.
This is a common method to reduce memory consumption and is utilized in many State of the Art
architectures today.

PyTorch XLA requires these weights to be tied/shared after moving the model to the XLA device.
To support this requirement, Lightning automatically finds these weights and ties them after
the modules are moved to the XLA device under the hood. It will ensure that the weights among
the modules are shared but not copied independently.

PyTorch Lightning has an inbuilt check which verifies that the model parameter lengths
match once the model is moved to the device. If the lengths do not match Lightning
throws a warning message.

Example:

.. code-block:: python

    from lightning.pytorch.core.module import LightningModule
    from torch import nn
    from lightning.pytorch.trainer.trainer import Trainer


    class WeightSharingModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(32, 10, bias=False)
            self.layer_2 = nn.Linear(10, 32, bias=False)
            self.layer_3 = nn.Linear(32, 10, bias=False)
            # Lightning automatically ties these weights after moving to the XLA device,
            # so all you need is to write the following just like on other accelerators.
            self.layer_3.weight = self.layer_1.weight

        def forward(self, x):
            x = self.layer_1(x)
            x = self.layer_2(x)
            x = self.layer_3(x)
            return x


    model = WeightSharingModule()
    trainer = Trainer(max_epochs=1, accelerator="tpu")

See `XLA Documentation <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#xla-tensor-quirks>`_

----

XLA
---
XLA is the library that interfaces PyTorch with the TPUs.
For more information check out `XLA <https://github.com/pytorch/xla>`_.

Guide for `troubleshooting XLA <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md>`_
