################
Fabric Arguments
################


accelerator
===========

Choose one of ``"cpu"``, ``"gpu"``, ``"tpu"``, ``"auto"``.

.. code-block:: python

    # CPU accelerator
    fabric = Fabric(accelerator="cpu")

    # Running with GPU Accelerator using 2 GPUs
    fabric = Fabric(devices=2, accelerator="gpu")

    # Running with TPU Accelerator using 8 TPU cores
    fabric = Fabric(devices=8, accelerator="tpu")

    # Running with GPU Accelerator using the DistributedDataParallel strategy
    fabric = Fabric(devices=4, accelerator="gpu", strategy="ddp")

The ``"auto"`` option recognizes the machine you are on and selects the available accelerator.

.. code-block:: python

    # If your machine has GPUs, it will use the GPU Accelerator
    fabric = Fabric(devices=2, accelerator="auto")


See also: :doc:`../fundamentals/accelerators`


strategy
========

Choose a training strategy: ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"ddp_find_unused_parameters_true"``, ``"xla"``, ``"deepspeed"``, ``"fsdp"``.

.. code-block:: python

    # Running with the DistributedDataParallel strategy on 4 GPUs
    fabric = Fabric(strategy="ddp", accelerator="gpu", devices=4)

    # Running with the DDP strategy with find unused parameters enabled on 4 GPUs
    fabric = Fabric(strategy="ddp_find_unused_parameters_true", accelerator="gpu", devices=4)

    # Running with the DDP Spawn strategy using 4 CPU processes
    fabric = Fabric(strategy="ddp_spawn", accelerator="cpu", devices=4)


Additionally, you can pass in your custom strategy by configuring additional parameters.

.. code-block:: python

    from lightning.fabric.strategies import DeepSpeedStrategy

    fabric = Fabric(strategy=DeepSpeedStrategy(stage=2), accelerator="gpu", devices=2)

See also: :doc:`../fundamentals/launch`


devices
=======

Configure the devices to run on. Can be of type:

- int: the number of devices (e.g., GPUs) to train on
- list of int: which device index (e.g., GPU ID) to train on (0-indexed)
- str: a string representation of one of the above

.. code-block:: python

    # default used by Fabric, i.e., use the CPU
    fabric = Fabric(devices=None)

    # equivalent
    fabric = Fabric(devices=0)

    # int: run on two GPUs
    fabric = Fabric(devices=2, accelerator="gpu")

    # list: run on the 2nd (idx 1) and 5th (idx 4) GPUs (by bus ordering)
    fabric = Fabric(devices=[1, 4], accelerator="gpu")
    fabric = Fabric(devices="1, 4", accelerator="gpu")  # equivalent

    # -1: run on all GPUs
    fabric = Fabric(devices=-1, accelerator="gpu")
    fabric = Fabric(devices="-1", accelerator="gpu")  # equivalent

See also: :doc:`../fundamentals/launch`


num_nodes
=========


The number of cluster nodes for distributed operation.

.. code-block:: python

    # Default used by Fabric
    fabric = Fabric(num_nodes=1)

    # Run on 8 nodes
    fabric = Fabric(num_nodes=8)


Learn more about :ref:`distributed multi-node training on clusters <Fabric Cluster>`.


precision
=========

There are two different techniques to set the mixed precision. "True" precision and "Mixed" precision.
For an extensive guide into their differences, please see: :doc:`../fundamentals/precision`

Fabric supports doing floating point operations in 64-bit precision ("double"), 32-bit precision ("full"), or 16-bit ("half") with both regular and `bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_).
This selected precision will have a direct impact in the performance and memory usage based on your hardware.
Automatic mixed precision settings are denoted by a ``"-mixed"`` suffix, while "true" precision settings have a ``"-true"`` suffix:

.. code-block:: python

    # Default used by the Fabric
    fabric = Fabric(precision="32-true", devices=1)

    # the same as:
    fabric = Fabric(precision="32", devices=1)

    # 16-bit mixed precision (model weights remain in torch.float32)
    fabric = Fabric(precision="16-mixed", devices=1)

    # 16-bit bfloat mixed precision (model weights remain in torch.float32)
    fabric = Fabric(precision="bf16-mixed", devices=1)

    # 8-bit mixed precision via TransformerEngine (model weights get cast to torch.bfloat16)
    fabric = Fabric(precision="transformer-engine", devices=1)

    # 16-bit precision (model weights get cast to torch.float16)
    fabric = Fabric(precision="16-true", devices=1)

    # 16-bit bfloat precision (model weights get cast to torch.bfloat16)
    fabric = Fabric(precision="bf16-true", devices=1)

    # 64-bit (double) precision (model weights get cast to torch.float64)
    fabric = Fabric(precision="64-true", devices=1)


Precision settings can also be enabled via the plugins argument (see section below on plugins).
An example is the weights quantization plugin Bitsandbytes for 4-bit and 8-bit:

.. code-block:: python

    from lightning.fabric.plugins import BitsandbytesPrecision

    precision = BitsandbytesPrecision(mode="nf4-dq", dtype=torch.bfloat16)
    fabric = Fabric(plugins=precision)


plugins
=======

Plugins allow you to connect arbitrary backends, precision libraries, clusters, etc. For example:
To define your own behavior, subclass the relevant class and pass it in. Here's an example linking up your own
:class:`~lightning.fabric.plugins.environments.cluster_environment.ClusterEnvironment`.

.. code-block:: python

    from lightning.fabric.plugins.environments import ClusterEnvironment


    class MyCluster(ClusterEnvironment):
        @property
        def main_address(self):
            return your_main_address

        @property
        def main_port(self):
            return your_main_port

        def world_size(self):
            return the_world_size


    fabric = Fabric(plugins=[MyCluster()], ...)


callbacks
=========

A callback class is a collection of methods that the training loop can call at a specific time, for example, at the end of an epoch.
Add callbacks to Fabric to inject logic into your training loop from an external callback class.

.. code-block:: python

    class MyCallback:
        def on_train_epoch_end(self, results):
            ...

You can then register this callback or multiple ones directly in Fabric:

.. code-block:: python

    fabric = Fabric(callbacks=[MyCallback()])


Then, in your training loop, you can call a hook by its name. Any callback objects that have this hook will execute it:

.. code-block:: python

    # Call any hook by name
    fabric.call("on_train_epoch_end", results={...})

See also: :doc:`../guide/callbacks`


loggers
=======

Attach one or several loggers/experiment trackers to Fabric for convenient metrics logging.

.. code-block:: python

    # Default used by Fabric; no loggers are active
    fabric = Fabric(loggers=[])

    # Log to a single logger
    fabric = Fabric(loggers=TensorBoardLogger(...))

    # Or multiple instances
    fabric = Fabric(loggers=[logger1, logger2, ...])

Anywhere in your training loop, you can log metrics to all loggers at once:

.. code-block:: python

    fabric.log("loss", loss)
    fabric.log_dict({"loss": loss, "accuracy": acc})


See also: :doc:`../guide/logging`
