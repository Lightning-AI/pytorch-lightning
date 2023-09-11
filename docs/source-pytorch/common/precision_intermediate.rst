:orphan:

.. _precision_intermediate:

##############################
N-Bit Precision (Intermediate)
##############################
**Audience:** Users looking to scale larger models or take advantage of optimized accelerators.

----

************************
What is Mixed Precision?
************************

PyTorch, like most deep learning frameworks, trains on 32-bit floating-point (FP32) arithmetic by default. However, many deep learning models do not require this to reach complete accuracy. By conducting
operations in half-precision format while keeping minimum information in single-precision to maintain as much information as possible in crucial areas of the network, mixed precision training delivers
significant computational speedup. Switching to mixed precision has resulted in considerable training speedups since the introduction of Tensor Cores in the Volta and Turing architectures. It combines
FP32 and lower-bit floating-points (such as FP16) to reduce memory footprint and increase performance during model training and evaluation. It accomplishes this by recognizing the steps that require
complete accuracy and employing a 32-bit floating-point for those steps only, while using a 16-bit floating-point for the rest. When compared to complete precision training, mixed precision training
delivers all of these benefits while ensuring that no task-specific accuracy is lost. [`2 <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html>`_].

.. note::

    In some cases, it is essential to remain in FP32 for numerical stability, so keep this in mind when using mixed precision.
    For example, when running scatter operations during the forward (such as torchpoint3d), computation must remain in FP32.

.. warning::

    Do not cast anything to other dtypes manually using ``torch.autocast`` or ``tensor.half()`` when using native precision because
    this can bring instability.

    .. code-block:: python

        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                outs = self(batch)

                a_float32 = torch.rand((8, 8), device=self.device, dtype=self.dtype)
                b_float32 = torch.rand((8, 4), device=self.device, dtype=self.dtype)

                # casting to float16 manually
                with torch.autocast(device_type=self.device.type):
                    c_float16 = torch.mm(a_float32, b_float32)
                    target = self.layer(c_float16.flatten()[None])

                # here outs is of type float32 and target is of type float16
                loss = torch.mm(target @ outs).float()
                return loss


        trainer = Trainer(accelerator="gpu", devices=1, precision=32)

----


************************
BFloat16 Mixed Precision
************************

.. warning::

    BFloat16 may not provide significant speedups or memory improvements or offer better numerical stability.
    For GPUs, the most significant benefits require `Ampere <https://en.wikipedia.org/wiki/Ampere_(microarchitecture)>`__ based GPUs or newer, such as A100s or 3090s.

BFloat16 Mixed precision is similar to FP16 mixed precision, however, it maintains more of the "dynamic range" that FP32 offers. This means it is able to improve numerical stability than FP16 mixed precision. For more information, see `this TPU performance blogpost <https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus>`__.

Under the hood, we use `torch.autocast <https://pytorch.org/docs/stable/amp.html>`__ with the dtype set to ``bfloat16``, with no gradient scaling.

.. testcode::
    :skipif: not torch.cuda.is_available()

    Trainer(accelerator="gpu", devices=1, precision="bf16-mixed")

It is also possible to use BFloat16 mixed precision on the CPU, relying on MKLDNN under the hood.

.. testcode::

    Trainer(precision="bf16-mixed")


----


*******************
True Half Precision
*******************

As mentioned before, for numerical stability mixed precision keeps the model weights in full float32 precision while casting only supported operations to lower bit precision.
However, in some cases it is indeed possible to train completely in half precision. Similarly, for inference the model weights can often be cast to half precision without a loss in accuracy (even when trained with mixed precision).

.. code-block:: python

    # Select FP16 precision
    trainer = Trainer(precision="16-true")
    trainer.fit(model)  # model gets cast to torch.float16

    # Select BF16 precision
    trainer = Trainer(precision="bf16-true")
    trainer.fit(model)  # model gets cast to torch.bfloat16

Tip: For faster initialization, you can create model parameters with the desired dtype directly on the device:

.. code-block:: python

    trainer = Trainer(precision="bf16-true")

    # init the model directly on the device and with parameters in half-precision
    with trainer.init_module():
        model = MyModel()

    trainer.fit(model)


See also: :doc:`../advanced/model_init`


----


*****************************************************
Float8 Mixed Precision via Nvidia's TransformerEngine
*****************************************************

`Transformer Engine <https://github.com/NVIDIA/TransformerEngine>`__ (TE) is a library for accelerating models on the
latest NVIDIA GPUs using 8-bit floating point (FP8) precision on Hopper GPUs, to provide better performance with lower
memory utilization in both training and inference. It offers improved performance over half precision with no degradation in accuracy.

Using TE requires replacing some of the layers in your model. Fabric automatically replaces the :class:`torch.nn.Linear`
and :class:`torch.nn.LayerNorm` layers in your model with their TE alternatives, however, TE also offers
`fused layers <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html>`__
to squeeze out all the possible performance. If Fabric detects that any layer has been replaced already, automatic
replacement is not done.

This plugin is a combination of "mixed" and "true" precision. The computation is downcasted to FP8 precision on the fly, but
the model and inputs can be kept in true full or half precision.

.. code-block:: python

    # Select 8bit mixed precision via TransformerEngine
    fabric = Trainer(precision="transformer-engine")

    # Customize the fp8 recipe or set a different base precision:
    from lightning.trainer.plugins import TransformerEnginePrecision

    recipe = {"fp8_format": "HYBRID", "amax_history_len": 16, "amax_compute_algo": "max"}
    precision = TransformerEnginePrecision(dtype=torch.bfloat16, recipe=recipe)
    fabric = Trainer(plugins=precision)


Under the hood, we use `transformer_engine.pytorch.fp8_autocast <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.fp8_autocast>`__ with the default fp8 recipe.

.. note::

    This requires `Hopper <https://en.wikipedia.org/wiki/Hopper_(microarchitecture)>`_ based GPUs or newer, such the H100.


----


***************
8-bit Optimizer
***************

It is possible to further reduce the memory usage of the optimizer states by using third-party libraries like `bitsandbytes <https://github.com/TimDettmers/bitsandbytes>`_.
You can configure it in your LightningModule by overriding ``configure_optimizers``.

.. code-block:: python

    import bitsandbytes as bnb


    # in your LightningModule, return the 8-bit optimizer
    def configure_optimizers(self):
        return bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995))
