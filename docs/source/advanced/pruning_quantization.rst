.. testsetup:: *

    import os
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule

.. _pruning_quantization:

########################
Pruning and Quantization
########################

Pruning and Quantization are tecniques to compress models for deployment, allowing memory and energy reduction without significant accuracy losses.

*******
Pruning
*******

.. warning ::
     Pruning is in beta and subject to change.

Pruning is a technique to optimize model memory, hardware, and energy requirements by eliminating some of the model weights. Pruning has been shown to achieve significant efficiency improvements while minimizing the drop in performance. The pruned model is smaller in size, more memory-efficient requires less energy and memory, and is faster to run with minimal accuracy drop.

TODO: when is it recomended?

To enable pruning during training in Lightning, simply pass in the :class:`~pytorch_lightning.callbacks.ModelPruning` callback to the Lighting Trainer (using native PyTorch pruning implementation under the hood).

This callback suports multiple pruning functions: pass any `torch.nn.utils.prune <https://pytorch.org/docs/stable/nn.html#utilities>`_ function as a string to select which weights to pruned (`random_unstructured <https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.random_unstructured.html#torch.nn.utils.prune.random_unstructured>`_, `RandomStructured <https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured>`_, etc) or implement your own by subclassing `BasePruningMethod <https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#extending-torch-nn-utils-prune-with-custom-pruning-functions>`_.

TODO: what do you have to set?

You can also set the pruning percentage, perform iterative pruning, apply the <lottery ticket hypothesis <https://arxiv.org/pdf/1803.03635.pdf>`_ and more!

.. code-block:: python


	from pytorch_lightning.callbacks import ModelPruning

	trainer = Trainer(callbacks=[ModelPruning("random_unstructured")])


************
Quantization
************

.. warning ::
     Quantization is in beta and subject to change.

Model quantization is another performance optimization technique allows speeding up inference and decreasing memory requirements by performing computations and storing tensors at lower bitwidths (such as INT8 or FLOAT16) than floating point precision. Quantization not only reduces the model size, but also speeds up loading since operations on fixpoint are faster than on floating-point. 

Quantization Aware Training (QAT) mimics the effects of quantization during training: all computations are carried out in floating points while training, simulating the effects of ints, and weights and activations are quantized into lower precision only once training is completed.

TODO: when is it recomended?

Lightning includes :class:`~pytorch_lightning.callbacks.QuantizationAwareTraining` callback (using PyTorch native quantization, read more `here <https://pytorch.org/docs/stable/quantization.html#quantization-aware-training>`_), which allows creating fully quantized models (compatible with torchscript).

To quantize your model, specify TODO(borda).

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

	qcb = QuantizationAwareTraining(
			# specification of quant estimation quaity
			observer_type='histogram',
			# specify which layers shall be merged together to increase efficiency
			modules_to_fuse=[(f'layer_{i}', f'layer_{i}a') for i in range(2)],
	)

	trainer = Trainer(callbacks=[qcb])
	trainer.fit(model, ...)

 You can also make your model compatible with all original input/outputs, in such case the model is wrapped in a shell with entry/exit layers.

 TODO(borda): add code example

