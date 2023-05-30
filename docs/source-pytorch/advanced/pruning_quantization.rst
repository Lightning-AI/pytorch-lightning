.. _pruning_quantization:

########################
Pruning and Quantization
########################

Pruning and Quantization are techniques to compress model size for deployment, allowing inference speed up and energy saving without significant accuracy losses.

*******
Pruning
*******

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

Pruning is a technique which focuses on eliminating some of the model weights to reduce the model size and decrease inference requirements.

Pruning has been shown to achieve significant efficiency improvements while minimizing the drop in model performance (prediction quality). Model pruning is recommended for cloud endpoints, deploying models on edge devices, or mobile inference (among others).

To enable pruning during training in Lightning, simply pass in the :class:`~lightning.pytorch.callbacks.ModelPruning` callback to the Lightning Trainer. PyTorch's native pruning implementation is used under the hood.

This callback supports multiple pruning functions: pass any `torch.nn.utils.prune <https://pytorch.org/docs/stable/nn.html#utilities>`_ function as a string to select which weights to prune (`random_unstructured <https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.random_unstructured.html#torch.nn.utils.prune.random_unstructured>`_, `RandomStructured <https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured>`_, etc) or implement your own by subclassing `BasePruningMethod <https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#extending-torch-nn-utils-prune-with-custom-pruning-functions>`_.

.. code-block:: python

    from lightning.pytorch.callbacks import ModelPruning

    # set the amount to be the fraction of parameters to prune
    trainer = Trainer(callbacks=[ModelPruning("l1_unstructured", amount=0.5)])

You can also perform iterative pruning, apply the `lottery ticket hypothesis <https://arxiv.org/abs/1803.03635>`__, and more!

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



Post-training Quantization
==========================

If you want to quantize a fine-tuned model with PTQ, it is recommended to adopt a third party API names Intel® Neural Compressor, read more :doc:`here <./post_training_quantization>`, which provides a convenient tool for accelerating the model inference speed on Intel CPUs and GPUs.
