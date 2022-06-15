:orphan:

.. _hpu_basics:

Accelerator: HPU training
=========================
**Audience:** Users looking to save money and run large models faster using single or multiple Gaudi devices.

----

What is an HPU?
---------------

`Habana® Gaudi® AI Processor (HPU) <https://habana.ai/>`__ training processors are built on a heterogeneous architecture with a cluster of fully programmable Tensor Processing Cores (TPC) along with its associated development tools and libraries, and a configurable Matrix Math engine.

The TPC core is a VLIW SIMD processor with an instruction set and hardware tailored to serve training workloads efficiently.
The Gaudi memory architecture includes on-die SRAM and local memories in each TPC and,
Gaudi is the first DL training processor that has integrated RDMA over Converged Ethernet (RoCE v2) engines on-chip.

On the software side, the PyTorch Habana bridge interfaces between the framework and SynapseAI software stack to enable the execution of deep learning models on the Habana Gaudi device.

Gaudi offers a substantial price/performance advantage -- so you get to do more deep learning training while spending less.

For more information, check out `Gaudi Architecture <https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Overview.html#gaudi-architecture>`__ and `Gaudi Developer Docs <https://developer.habana.ai>`__.

----

Run on 1 Gaudi
--------------

To enable PyTorch Lightning to utilize the HPU accelerator, simply provide ``accelerator="hpu"`` parameter to the Trainer class.

.. code-block:: python

    trainer = Trainer(accelerator="hpu", devices=1)

----

Run on multiple Gaudis
----------------------
The ``devices=8`` and ``accelerator="hpu"`` parameters to the Trainer class enables the Habana accelerator for distributed training with 8 Gaudis.
It uses :class:`~pytorch_lightning.strategies.hpu_parallel.HPUParallelStrategy` internally which is based on DDP strategy with the addition of Habana's collective communication library (HCCL) to support scale-up within a node and scale-out across multiple nodes.

.. code-block:: python

    trainer = Trainer(devices=8, accelerator="hpu")

----

Select Gaudis automatically
---------------------------

Lightning can automatically detect the number of Gaudi devices to run on. This setting is enabled by default if the devices argument is missing.

.. code-block:: python

    # equivalent
    trainer = Trainer(accelerator="hpu")
    trainer = Trainer(accelerator="hpu", devices="auto")

----

How to access HPUs
------------------

To use HPUs, you must have access to a system with HPU devices.

AWS
^^^
You can either use `Gaudi-based AWS EC2 DL1 instances <https://aws.amazon.com/ec2/instance-types/dl1/>`__ or `Supermicro X12 Gaudi server <https://www.supermicro.com/en/solutions/habana-gaudi>`__ to get access to HPUs.

Check out the `Get Started Guide with AWS and Habana <https://docs.habana.ai/en/latest/AWS_EC2_Getting_Started/AWS_EC2_Getting_Started.html>`__.

----

.. _known-limitations_hpu:

Known limitations
-----------------

* Multiple optimizers are not supported.
* `Habana dataloader <https://docs.habana.ai/en/latest/PyTorch_User_Guide/PyTorch_User_Guide.html#habana-data-loader>`__ is not supported.
* :class:`~pytorch_lightning.callbacks.device_stats_monitor.DeviceStatsMonitor` is not supported.
* :func:`torch.inference_mode` is not supported
