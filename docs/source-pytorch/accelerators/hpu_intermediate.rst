:orphan:

.. _hpu_intermediate:

Accelerator: HPU training
=========================
**Audience:** Gaudi chip users looking to save memory and scale models with mixed-precision training.

----

Enable Mixed Precision
----------------------

Lightning also allows mixed precision training with HPUs.
By default, HPU training will use 32-bit precision. To enable mixed precision, set the ``precision`` flag.

.. code-block:: python

    trainer = Trainer(devices=1, accelerator="hpu", precision=16)

----

Customize Mixed Precision
-------------------------

Internally, :class:`~pytorch_lightning.plugins.precision.hpu.HPUPrecisionPlugin` uses the Habana Mixed Precision (HMP) package to enable mixed precision training.

You can execute the ops in FP32 or BF16 precision. The HMP package modifies the Python operators to add the appropriate cast operations for the arguments before execution.
The default settings enable users to enable mixed precision training with minimal code easily.

In addition to the default settings in HMP, users also have the option of overriding these defaults and providing their
BF16 and FP32 operator lists by passing them as parameter to :class:`~pytorch_lightning.plugins.precision.hpu.HPUPrecisionPlugin`.

The below snippet shows an example model using MNIST with a single Habana Gaudi device and making use of HMP by overriding the default parameters.
This enables advanced users to provide their own BF16 and FP32 operator list instead of using the HMP defaults.

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.plugins import HPUPrecisionPlugin

    # Initialize a trainer with HPU accelerator for HPU strategy for single device,
    # with mixed precision using overidden HMP settings
    trainer = pl.Trainer(
        accelerator="hpu",
        devices=1,
        # Optional Habana mixed precision params to be set
        # Checkout `examples/pl_hpu/ops_bf16_mnist.txt` for the format
        plugins=[
            HPUPrecisionPlugin(
                precision=16,
                opt_level="O1",
                verbose=False,
                bf16_file_path="ops_bf16_mnist.txt",
                fp32_file_path="ops_fp32_mnist.txt",
            )
        ],
    )

    # Init our model
    model = LitClassifier()
    # Init the data
    dm = MNISTDataModule(batch_size=batch_size)

    # Train the model ⚡
    trainer.fit(model, datamodule=dm)

For more details, please refer to `PyTorch Mixed Precision Training on Gaudi <https://docs.habana.ai/en/latest/PyTorch/PyTorch_Mixed_Precision/PT_Mixed_Precision.html>`__.

----

Enabling DeviceStatsMonitor with HPUs
----------------------------------------

:class:`~pytorch_lightning.callbacks.device_stats_monitor.DeviceStatsMonitor` is a callback that automatically monitors and logs device stats during the training stage.
This callback can be passed for training with HPUs. It returns a map of the following metrics with their values in bytes of type uint64:

- **Limit**: amount of total memory on HPU device.
- **InUse**: amount of allocated memory at any instance.
- **MaxInUse**: amount of total active memory allocated.
- **NumAllocs**: number of allocations.
- **NumFrees**: number of freed chunks.
- **ActiveAllocs**: number of active allocations.
- **MaxAllocSize**: maximum allocated size.
- **TotalSystemAllocs**: total number of system allocations.
- **TotalSystemFrees**: total number of system frees.
- **TotalActiveAllocs**: total number of active allocations.

The below snippet shows how DeviceStatsMonitor can be enabled.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import DeviceStatsMonitor

    device_stats = DeviceStatsMonitor()
    trainer = Trainer(accelerator="hpu", callbacks=[device_stats])

For more details, please refer to `Memory Stats APIs <https://docs.habana.ai/en/v1.5.0/PyTorch/PyTorch_User_Guide/Python_Packages.html#memory-stats-apis>`__.

----

Using HPUDataModule
-----------------------

``HPUDataModule`` class is a wrapper around the ``LightningDataModule`` class. It makes working with custom models easier on HPU devices.
It uses HabanaDataloader for training, testing, and validation of user-provided models. Currently, it only supports the ``Imagenet`` dataset.

Here's an example of how to use the ``HPUDataModule``:

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.utilities.hpu_datamodule import HPUDataModule

    train_dir = "./path/to/train/data"
    val_dir = "./path/to/val/data"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    val_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]

    data_module = HPUDataModule(
        train_dir,
        val_dir,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=8,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    # Initialize a trainer
    trainer = pl.Trainer(devices=1, accelerator="hpu", max_epochs=1, max_steps=2)

    # Init our model
    model = RN50Module()  # Or any other model to be defined by user

    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)

A working example can be found at ``examples/pl_hpu/hpu_datamodule_sample.py``.
For more details refer to `Habana dataloader <https://docs.habana.ai/en/latest/PyTorch_User_Guide/PyTorch_User_Guide.html#habana-data-loader>`__.
