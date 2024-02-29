################################
Save memory with mixed precision
################################

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric/animations/precision.mp4
    :width: 800
    :autoplay:
    :loop:
    :muted:
    :nocontrols:


************************
What is Mixed Precision?
************************

Like most deep learning frameworks, PyTorch runs on 32-bit floating-point (FP32) arithmetic by default.
However, many deep learning models do not require this to reach complete accuracy during training.
Mixed precision training delivers significant computational speedup by conducting operations in half-precision while keeping minimum information in single-precision to maintain as much information as possible in crucial areas of the network.
Switching to mixed precision has resulted in considerable training speedups since the introduction of Tensor Cores in the Volta and Turing architectures.
It combines FP32 and lower-bit floating points (such as FP16) to reduce memory footprint and increase performance during model training and evaluation.
It accomplishes this by recognizing the steps that require complete accuracy and employing a 32-bit floating point for those steps only while using a 16-bit floating point for the rest.
Compared to complete precision training, mixed precision training delivers all these benefits while ensuring no task-specific accuracy is lost `[1] <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html>`_.

This is how you select the precision in Fabric:

.. code-block:: python

    from lightning.fabric import Fabric

    # This is the default
    fabric = Fabric(precision="32-true")

    # Also FP32 (legacy)
    fabric = Fabric(precision=32)

    # FP32 as well (legacy)
    fabric = Fabric(precision="32")

    # Float16 mixed precision
    fabric = Fabric(precision="16-mixed")

    # Float16 true half precision
    fabric = Fabric(precision="16-true")

    # BFloat16 mixed precision (Volta GPUs and later)
    fabric = Fabric(precision="bf16-mixed")

    # BFloat16 true half precision (Volta GPUs and later)
    fabric = Fabric(precision="bf16-true")

    # 8-bit mixed precision via TransformerEngine (Hopper GPUs and later)
    fabric = Fabric(precision="transformer-engine")

    # Double precision
    fabric = Fabric(precision="64-true")

    # Or (legacy)
    fabric = Fabric(precision="64")

    # Or (legacy)
    fabric = Fabric(precision=64)


The same values can also be set through the :doc:`command line interface <launch>`:

.. code-block:: bash

    fabric run train.py --precision=bf16-mixed


.. note::

    In some cases, it is essential to remain in FP32 for numerical stability, so keep this in mind when using mixed precision.
    For example, when running scatter operations during the forward (such as torchpoint3d), the computation must remain in FP32.


----


********************
FP16 Mixed Precision
********************

In most cases, mixed precision uses FP16.
Supported `PyTorch operations <https://pytorch.org/docs/stable/amp.html#op-specific-behavior>`_ automatically run in FP16, saving memory and improving throughput on the supported accelerators.
Since computation happens in FP16, which has a very limited "dynamic range", there is a chance of numerical instability during training.
This is handled internally by a dynamic grad scaler which skips invalid steps and adjusts the scaler to ensure subsequent steps fall within a finite range.
For more information `see the autocast docs <https://pytorch.org/docs/stable/amp.html#gradient-scaling>`_.

This is how you enable FP16 in Fabric:

.. code-block:: python

    # Select FP16 mixed precision
    fabric = Fabric(precision="16-mixed")

.. note::

    When using TPUs, setting ``precision="16-mixed"`` will enable bfloat16 based mixed precision, the only supported half-precision type on TPUs.


----


************************
BFloat16 Mixed Precision
************************

BFloat16 Mixed precision is similar to FP16 mixed precision. However, it maintains more of the "dynamic range" that FP32 offers.
This means it can improve numerical stability than FP16 mixed precision.
For more information, see `this TPU performance blog post <https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus>`_.

.. code-block:: python

    # Select BF16 precision
    fabric = Fabric(precision="bf16-mixed")


Under the hood, we use `torch.autocast <https://pytorch.org/docs/stable/amp.html>`__ with the dtype set to ``bfloat16``, with no gradient scaling.
It is also possible to use BFloat16 mixed precision on the CPU, relying on MKLDNN.

.. note::

    BFloat16 may not provide significant speedups or memory improvements, offering better numerical stability.
    For GPUs, the most significant benefits require `Ampere <https://en.wikipedia.org/wiki/Ampere_(microarchitecture)>`_ based GPUs or newer, such as A100s or 3090s.


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

    # Select 8bit mixed precision via TransformerEngine, with model weights in bfloat16
    fabric = Fabric(precision="transformer-engine")

    # Select 8bit mixed precision via TransformerEngine, with model weights in float16
    fabric = Fabric(precision="transformer-engine-float16")

    # Customize the fp8 recipe or set a different base precision:
    from lightning.fabric.plugins import TransformerEnginePrecision

    recipe = {"fp8_format": "HYBRID", "amax_history_len": 16, "amax_compute_algo": "max"}
    precision = TransformerEnginePrecision(weights_dtype=torch.bfloat16, recipe=recipe)
    fabric = Fabric(plugins=precision)


Under the hood, we use `transformer_engine.pytorch.fp8_autocast <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.fp8_autocast>`__ with the default fp8 recipe.

.. note::

    This requires `Hopper <https://en.wikipedia.org/wiki/Hopper_(microarchitecture)>`_ based GPUs or newer, such the H100.


----


*******************
True Half Precision
*******************

As mentioned before, for numerical stability mixed precision keeps the model weights in full float32 precision while casting only supported operations to lower bit precision.
However, in some cases it is indeed possible to train completely in half precision. Similarly, for inference the model weights can often be cast to half precision without a loss in accuracy (even when trained with mixed precision).

.. code-block:: python

    # Select FP16 precision
    fabric = Fabric(precision="16-true")
    model = MyModel()
    model = fabric.setup(model)  # model gets cast to torch.float16

    # Select BF16 precision
    fabric = Fabric(precision="bf16-true")
    model = MyModel()
    model = fabric.setup(model)  # model gets cast to torch.bfloat16

Tip: For faster initialization, you can create model parameters with the desired dtype directly on the device:

.. code-block:: python

    fabric = Fabric(precision="bf16-true")

    # init the model directly on the device and with parameters in half-precision
    with fabric.init_module():
        model = MyModel()

    model = fabric.setup(model)


See also: :doc:`../advanced/model_init`


----


*****************************
Quantization via Bitsandbytes
*****************************

`bitsandbytes <https://github.com/TimDettmers/bitsandbytes>`__ (BNB) is a library that supports quantizing :class:`torch.nn.Linear` weights.

Both 4-bit (`paper reference <https://arxiv.org/abs/2305.14314v1>`__) and 8-bit (`paper reference <https://arxiv.org/abs/2110.02861>`__) quantization is supported.
Specifically, we support the following modes:

* **nf4**: Uses the normalized float 4-bit data type. This is recommended over "fp4" based on the paper's experimental results and theoretical analysis.
* **nf4-dq**: "dq" stands for "Double Quantization" which reduces the average memory footprint by quantizing the quantization constants. In average, this amounts to about 0.37 bits per parameter (approximately 3 GB for a 65B model).
* **fp4**: Uses regular float 4-bit data type.
* **fp4-dq**: "dq" stands for "Double Quantization" which reduces the average memory footprint by quantizing the quantization constants. In average, this amounts to about 0.37 bits per parameter (approximately 3 GB for a 65B model).
* **int8**: Uses unsigned int8 data type.
* **int8-training**: Meant for int8 activations with fp16 precision weights.

While these techniques store weights in 4 or 8 bit, the computation still happens in 16 or 32-bit (float16, bfloat16, float32).
This is configurable via the dtype argument in the plugin.
If your model weights can fit on a single device with 16 bit precision, it's recommended that this plugin is not used as it will slow down training.

Quantizing the model will dramatically reduce the weight's memory requirements but may have a negative impact on the model's performance or runtime.

The :class:`~lightning.fabric.plugins.precision.bitsandbytes.BitsandbytesPrecision` automatically replaces the :class:`torch.nn.Linear` layers in your model with their BNB alternatives.

.. code-block:: python

    from lightning.fabric.plugins import BitsandbytesPrecision

    # this will pick out the compute dtype automatically, by default `bfloat16`
    precision = BitsandbytesPrecision(mode="nf4-dq")
    fabric = Fabric(plugins=precision)

    # Customize the dtype, or ignore some modules
    precision = BitsandbytesPrecision(mode="int8-training", dtype=torch.float16, ignore_modules={"lm_head"})
    fabric = Fabric(plugins=precision)

    model = MyModel()
    model = fabric.setup(model)


You can also directly initialize the model with the quantized layers if you are not setting any ``ignore_modules=...`` by
initializing your model under the :meth:`~lightning.fabric.fabric.Fabric.init_module` context manager.


.. note::

    Only supports CUDA devices and the Linux operating system. Windows users should use
    `WSL2 <https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl>`__.


This plugin does not take care of replacing your optimizer with an 8-bit optimizer e.g. ``bitsandbytes.optim.Adam8bit``.
You might want to do this for extra memory savings.

.. code-block:: python

    import bitsandbytes as bnb

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995))

    # (optional) force embedding layers to use 32 bit for numerical stability
    # https://github.com/huggingface/transformers/issues/14819#issuecomment-1003445038
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            bnb.optim.GlobalOptimManager.get_instance().register_module_override(module, "weight", {"optim_bits": 32})


----


*********************
True Double Precision
*********************

For certain scientific computations, 64-bit precision enables more accurate models. However, doubling the precision from 32 to 64 bit also doubles the memory requirements.

.. code-block:: python

    # Select FP64 precision
    fabric = Fabric(precision="64-true")
    model = MyModel()
    model = fabric.setup(model)  # model gets cast to torch.float64

Since in deep learning, memory is always a bottleneck, especially when dealing with a large volume of data and with limited resources.
It is recommended using single precision for better speed. Although you can still use it if you want for your particular use-case.

When working with complex numbers, instantiation of complex tensors should be done under the
:meth:`~lightning.fabric.fabric.Fabric.init_module` context manager so that the `complex128` dtype
is properly selected.

.. code-block:: python

    fabric = Fabric(precision="64-true")

    # init the model directly on the device and with parameters in full-precision
    with fabric.init_module():
        model = MyModel()

    model = fabric.setup(model)


----


************************************
Control where precision gets applied
************************************

Fabric automatically casts the data type and operations in the ``forward`` of your model:

.. code-block:: python

    fabric = Fabric(precision="bf16-mixed")

    model = ...
    optimizer = ...

    # Here, Fabric sets up the `model.forward` for precision auto-casting
    model, optimizer = fabric.setup(model, optimizer)

    # Precision casting gets handled in your forward, no code changes required
    output = model.forward(input)

    # Precision does NOT get applied here (only in forward)
    loss = loss_function(output, target)

If you want to enable operations in lower bit-precision **outside** your model's ``forward()``, you can use the :meth:`~lightning.fabric.fabric.Fabric.autocast` context manager:

.. code-block:: python

    # Precision now gets also handled in this part of the code:
    with fabric.autocast():
        loss = loss_function(output, target)
