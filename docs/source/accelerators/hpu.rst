.. _hpu:

Habana Gaudi AI Processor
=========================

Habana® Gaudi® AI training processors have been architected from the ground up and optimized for deep learning training efficiency.
Gaudi offers substantial price/performance advantage -- so you get to do more deep learning training while spending less.

You can use either the Gaudi-based AWS EC2 DL1 instances `<https://aws.amazon.com/ec2/instance-types/dl1/>` or the Supermicro X12 Gaudi server `< https://www.supermicro.com/en/solutions/habana-gaudi>`

Habana’s SynapseAI® software suite is optimized for building and training deep learning models using TensorFlow and PyTorch frameworks.  Gaudi is referred to as the Habana Processing Unit (HPU).
With SynapseAI, we aim to make training workloads on Gaudi easy, whether you're developing from scratch or migrating existing workloads.  Lightning supports running on HPUs.
For more information, check out `<https://developer.habana.ai>` and `<https://habana.ai/>`_.

PyTorch Lightning With Gaudi HPU
================================

Lightning supports training on a single HPU device or 8 HPU devices with the following plugins


.. _hpu_accelerator:

HPU accelerator
---------------

The :code:`devices=1` with :code:`accelerator="hpu"` parameters in the trainer class enables the Habana backend.


.. _single_device_strategy:

Training on Single HPU
----------------------

The :code:`devices=1` and :code:`accelerator="hpu"` with :code:`strategy=SingleHPUStrategy(device=torch.device("hpu"))` parameter in the trainer class enables the Habana backend for single Gaudi training.


.. _parallel_device_strategy:

Distributed Training
---------------------


The :code:`devices=8` and :code:`accelerator="hpu"` with :code:`strategy=HPUParallelStrategy( parallel_devices=[torch.device("hpu")] * devices)`  parameter in the trainer class enables the Habana backend for distributed training with 8 Gaudis.

The Habana parallel device strategy is based on DDP strategy with the addition of  Habana's collective communication library (HCCL) to support scale-up within a node and scale-out across multiple nodes.


.. _mixed_precision_plugin:

Mixed Precision Plugin
----------------------

The :code:`precision=16` and a :code:`hmp_params` parameter in the trainer class enables the Habana plugin for mixed precision using the Habana Mixed Precision (HMP) package.

You can execute the ops in FP32 or BF16 precision. The HMP package modifies the python operators to add the appropriate cast operations for the arguments before execution.
The default settings enable users to easily enable mixed precision training with minimal code.

In addition to the default settings in HMP,  users also have the option of overriding these defaults and providing your own BF16 and FP32 operator lists

For more details, please refer `<https://docs.habana.ai/en/master/PyTorch/PyTorch_User_Guide/PT_Mixed_Precision.html#pytorch-mixed-precision-training>`_.


.. _pytorch_lightning_examples:

Getting Started with Lightning on Gaudi
=======================================

This section describes how to train models using Habana PyTorch with Gaudi.

More Lightning HPU examples can be found in  pl_examples (`<https://github.com/PyTorchLightning/pytorch-lightning/pl_examples/hpu_examples/ >`)

Enabling Lightning with Single Gaudi HPU
----------------------------------------

The below snippet shows an example model using MNIST with single Habana Gaudi.

.. code-block:: python

    import habana_frameworks.torch.core as htcore


    class LitClassifier(pl.LightningModule):
        def __init__(self):
            super(LitClassifier, self).__init__()

        ...


    # Init our model
    model = LitClassifier()

    # Init DataLoader from MNIST Dataset
    dm = MNISTDataModule(batch_size=batch_size)

    ...

    num_hpus = 1

    # enable HPU strategy for single device, with mixed precision using default HMP settings
    hpustrat_1 = SingleHPUStrategy(device=torch.device("hpu"), precision_plugin=HPUPrecisionPlugin(precision=16))

    # Initialize a trainer with 1 HPU accelerator
    trainer = pl.Trainer(accelerator="hpu", devices=num_hpus, strategy=hpustrat_1)

    # Train the model ⚡
    trainer.fit(model, datamodule=dm)


Enabling Lightning with 8 Gaudi HPUs (distributed)
--------------------------------------------------

The below snippet shows an example model using MNIST with 8 Habana Gaudis.

.. code-block:: python

    import habana_frameworks.torch.core as htcore


    class LitClassifier(pl.LightningModule):
        def __init__(self):
            super(LitClassifier, self).__init__()

        ...


    # Init our model
    model = LitClassifier()

    # Init DataLoader from MNIST Dataset
    dm = MNISTDataModule(batch_size=batch_size)

    ...

    num_hpus = 8

    # setup parallel strategy for 8 HPU's
    hpustrat_8 = HPUParallelStrategy(
        parallel_devices=[torch.device("hpu")] * num_hpus,
        precision_plugin=HPUPrecisionPlugin(precision=16),
    )

    # Initialize a trainer with 1 HPU accelerator
    trainer = pl.Trainer(accelerator="hpu", devices=num_hpus, strategy=hpustrat_8)

    # Train the model ⚡
    trainer.fit(model, datamodule=dm)


Enabling Mixed Precision Options
--------------------------------

The below snippet shows an example model using MNIST with single Habana Gaudi and making use of HMP by overriding the default parameters.
This enables advanced users to provide their own bf16 and fp32 operator list instead of using the HMP defaults.

.. code-block:: python

    import habana_frameworks.torch.core as htcore


    class LitClassifier(pl.LightningModule):
        def __init__(self):
            super(LitClassifier, self).__init__()

        ...


    # Init our model
    model = LitClassifier()

    # Init DataLoader from MNIST Dataset
    dm = MNISTDataModule(batch_size=batch_size)

    ...

    num_hpus = 1

    # Optional Habana mixed precision params to be set
    hmp_keys = ["level", "verbose", "bf16_ops", "fp32_ops"]
    hmp_params = dict.fromkeys(hmp_keys)
    hmp_params["level"] = "O1"
    hmp_params["verbose"] = False
    hmp_params["bf16_ops"] = "ops_bf16_mnist.txt"
    hmp_params["fp32_ops"] = "ops_fp32_mnist.txt"

    # enable HPU strategy for single device, with mixed precision using overidden HMP settings
    hpustrat_1 = SingleHPUStrategy(
        device=torch.device("hpu"), precision_plugin=HPUPrecisionPlugin(precision=16, hmp_params=hmp_params)
    )

    # Initialize a trainer with 1 HPU accelerator
    trainer = pl.Trainer(accelerator="hpu", devices=num_hpus, strategy=hpustrat_1)

    # Train the model ⚡
    trainer.fit(model, datamodule=dm)


.. _known-limitations:

Known limitations
-----------------

* Habana dataloader is not supported
* Device stats monitoring is not supported
