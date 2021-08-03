.. testsetup:: *

    import os
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule

.. _pruning_quantization:

########################
Pruning and Quantization
########################

Pruning and Quantization are techniques to compress model size for deployment, allowing inference speed up and energy saving without significant accuracy losses.

*******
Pruning
*******

.. warning::

     Pruning is in beta and subject to change.

Pruning is a technique which focuses on eliminating some of the model weights to reduce the model size and decrease inference requirements.

Pruning has been shown to achieve significant efficiency improvements while minimizing the drop in model performance (prediction quality). Model pruning is recommended for cloud endpoints, deploying models on edge devices, or mobile inference (among others).

To enable pruning during training in Lightning, simply pass in the :class:`~pytorch_lightning.callbacks.ModelPruning` callback to the Lightning Trainer. PyTorch's native pruning implementation is used under the hood.

This callback supports multiple pruning functions: pass any `torch.nn.utils.prune <https://pytorch.org/docs/stable/nn.html#utilities>`_ function as a string to select which weights to prune (`random_unstructured <https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.random_unstructured.html#torch.nn.utils.prune.random_unstructured>`_, `RandomStructured <https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured>`_, etc) or implement your own by subclassing `BasePruningMethod <https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#extending-torch-nn-utils-prune-with-custom-pruning-functions>`_.

.. code-block:: python

    from pytorch_lightning.callbacks import ModelPruning

    # set the amount to be the fraction of parameters to prune
    trainer = Trainer(callbacks=[ModelPruning("l1_unstructured", amount=0.5)])

You can also perform iterative pruning, apply the `lottery ticket hypothesis <https://arxiv.org/pdf/1803.03635.pdf>`__, and more!

.. code-block:: python

    def compute_amount(epoch):
        # the sum of all returned values need to be smaller than 1
        if epoch == 10:
            return 0.5

        elif epoch == 50:
            return 0.25

        elif 75 < epoch < 99:
            return 0.01


    # the amount can be also be a callable
    trainer = Trainer(callbacks=[ModelPruning("l1_unstructured", amount=compute_amount)])


************
Quantization
************

.. warning ::
     Quantization is in beta and subject to change.

Model quantization is another performance optimization technique that allows speeding up inference and decreasing memory requirements by performing computations and storing tensors at lower bitwidths (such as INT8 or FLOAT16) than floating-point precision. This is particularly beneficial during model deployment.

Quantization Aware Training (QAT) mimics the effects of quantization during training: The computations are carried-out in floating-point precision but the subsequent quantization effect is taken into account. The weights and activations are quantized into lower precision only for inference, when training is completed.

Quantization is useful when it is required to serve large models on machines with limited memory, or when there's a need to switch between models and reducing the I/O time is important. For example, switching between monolingual speech recognition models across multiple languages.

Lightning includes :class:`~pytorch_lightning.callbacks.QuantizationAwareTraining` callback (using PyTorch's native quantization, read more `here <https://pytorch.org/docs/stable/quantization.html#quantization-aware-training>`__), which allows creating fully quantized models (compatible with torchscript).

.. code-block:: python

    from pytorch_lightning.callbacks import QuantizationAwareTraining


    class RegressionModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer_0 = nn.Linear(16, 64)
            self.layer_0a = torch.nn.ReLU()
            self.layer_1 = nn.Linear(64, 64)
            self.layer_1a = torch.nn.ReLU()
            self.layer_end = nn.Linear(64, 1)

        def forward(self, x):
            x = self.layer_0(x)
            x = self.layer_0a(x)
            x = self.layer_1(x)
            x = self.layer_1a(x)
            x = self.layer_end(x)
            return x


    trainer = Trainer(callbacks=[QuantizationAwareTraining()])
    qmodel = RegressionModel()
    trainer.fit(qmodel, ...)

    batch = iter(my_dataloader()).next()
    qmodel(qmodel.quant(batch[0]))

    tsmodel = qmodel.to_torchscript()
    tsmodel(tsmodel.quant(batch[0]))

You can further customize the callback:

.. code-block:: python


    qcb = QuantizationAwareTraining(
        # specification of quant estimation quality
        observer_type="histogram",
        # specify which layers shall be merged together to increase efficiency
        modules_to_fuse=[(f"layer_{i}", f"layer_{i}a") for i in range(2)],
        # make your model compatible with all original input/outputs, in such case the model is wrapped in a shell with entry/exit layers.
        input_compatible=True,
    )

    batch = iter(my_dataloader()).next()
    qmodel(batch[0])
