.. _hpu:

Habana Gaudi AI Processor (HPU)
===============================

Lightning supports `Habana Gaudi AI Processor (HPU) <https://habana.ai/>`__, for accelerating Deep Learning training workloads.

HPU terminology
---------------

Habana® Gaudi® AI training processors are built on a heterogeneous architecture with a cluster of fully programmable Tensor Processing Cores (TPC) along with its associated development tools and libraries, and a configurable Matrix Math engine.

The TPC core is a VLIW SIMD processor with instruction set and hardware tailored to serve training workloads efficiently.
The Gaudi memory architecture includes on-die SRAM and local memories in each TPC and,
Gaudi is the first DL training processor that has integrated RDMA over Converged Ethernet (RoCE v2) engines on-chip.

On the software side, the PyTorch Habana bridge interfaces between the framework and SynapseAI software stack to enable the execution of deep learning models on the Habana Gaudi device.

Gaudi offers substantial price/performance advantage -- so you get to do more deep learning training while spending less.

For more information, check out `Gaudi Architecture <https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Overview.html#gaudi-architecture>`__ and `Gaudi Developer Docs <https://developer.habana.ai>`__.

How to access HPUs
------------------

To use HPUs, you must have access to a system with HPU devices.
You can either use `Gaudi-based AWS EC2 DL1 instances <https://aws.amazon.com/ec2/instance-types/dl1/>`__ or `Supermicro X12 Gaudi server <https://www.supermicro.com/en/solutions/habana-gaudi>`__ to get access to HPUs.

Checkout the `Getting Started Guide with AWS and Habana <https://docs.habana.ai/en/latest/AWS_EC2_Getting_Started/AWS_EC2_Getting_Started.html>`__.

Training with HPUs
------------------

To enable PyTorch Lightning to utilize the HPU accelerator, simply provide ``accelerator="hpu"`` parameter to the Trainer class.

.. code-block:: python

    trainer = Trainer(accelerator="hpu")

Passing ``devices=1`` and ``accelerator="hpu"`` to the Trainer class enables the Habana accelerator for single Gaudi training.

.. code-block:: python

    trainer = Trainer(devices=1, accelerator="hpu")

The ``devices=8`` and ``accelerator="hpu"`` parameters to the Trainer class enables the Habana accelerator for distributed training with 8 Gaudis.
It uses :class:`~pytorch_lightning.strategies.HPUParallelStrategy` internally which is based on DDP strategy with the addition of Habana's collective communication library (HCCL) to support scale-up within a node and scale-out across multiple nodes.

.. code-block:: python

    trainer = Trainer(devices=8, accelerator="hpu")

.. note::
    If the ``devices`` flag is not defined, it will assume ``devices`` to be ``"auto"`` and fetch the :meth:`~pytorch_lightning.accelerators.HPUAccelerator.auto_device_count`
    from :class:`~pytorch_lightning.accelerators.HPUAccelerator`.


Mixed Precision Plugin
----------------------

Lightning also allows mixed precision training with HPUs.
By default, HPU training will use 32-bit precision. To enable mixed precision, set the ``precision`` flag.

In addition to the default settings in HMP, users also have the option of overriding these defaults and providing their own BF16 and FP32 operator lists using the ``plugins`` parameter of Trainer class.
HPU's precision plugin is realised using ``HPUPrecisionPlugin``. The ``hmp_params`` parameter with this plugin is used to override the default operator list. An example can be found in the subsequent section.

.. code-block:: python

    trainer = Trainer(devices=1, accelerator="hpu", precision=16)


Enabling Mixed Precision Options
--------------------------------

Internally, :class:`~pytorch_lightning.plugins.precision.HPUPrecisionPlugin` uses the Habana Mixed Precision (HMP) package to enable mixed precision training.

You can execute the ops in FP32 or BF16 precision. The HMP package modifies the python operators to add the appropriate cast operations for the arguments before execution.
The default settings enable users to easily enable mixed precision training with minimal code.

In addition to the default settings in HMP, users also have the option of overriding these defaults and providing their own BF16 and FP32 operator lists by passing it
to the ``hmp_params`` parameter of :class:`~pytorch_lightning.plugins.precision.HPUPrecisionPlugin`.

The below snippet shows an example model using MNIST with single Habana Gaudi and making use of HMP by overriding the default parameters.
This enables advanced users to provide their own BF16 and FP32 operator list instead of using the HMP defaults.

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.plugins import HPUPrecisionPlugin


    class LitClassifier(pl.LightningModule):
        def __init__(self):
            super(LitClassifier, self).__init__()

        ...


    # Init our model
    model = LitClassifier()

    # Init DataLoader from MNIST Dataset
    dm = MNISTDataModule(batch_size=batch_size)

    ...

    # Optional Habana mixed precision params to be set
    hmp_keys = ["level", "verbose", "bf16_ops", "fp32_ops"]
    hmp_params = dict.fromkeys(hmp_keys)
    hmp_params["level"] = "O1"
    hmp_params["verbose"] = False
    hmp_params["bf16_ops"] = "ops_bf16_mnist.txt"
    hmp_params["fp32_ops"] = "ops_fp32_mnist.txt"

    # Initialize a trainer with HPU accelerator for HPU strategy for single device,
    # with mixed precision using overidden HMP settings
    trainer = pl.Trainer(accelerator="hpu", devices=1, plugins=[HPUPrecisionPlugin(precision=16, hmp_params=hmp_params)])

    # Train the model ⚡
    trainer.fit(model, datamodule=dm)

For more details, please refer to `PyTorch Mixed Precision Training on Gaudi <https://docs.habana.ai/en/latest/PyTorch_User_Guide/PyTorch_User_Guide.html#pytorch-mixed-precision-training-on-gaudi>`__.

----------------

.. _known-limitations_hpu:

Known limitations
-----------------

* Multiple optimizers are not supported.
* `Habana dataloader <https://docs.habana.ai/en/latest/PyTorch_User_Guide/PyTorch_User_Guide.html#habana-data-loader>`__ is not supported.
* :class:`~pytorch_lightning.callbacks.DeviceStatsMonitor` is not supported.
