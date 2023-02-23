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

Scale-out on Gaudis
-------------------

To train a Lightning model using multiple HPU nodes, set the ``num_nodes`` parameter with the available nodes in the ``Trainer`` class.

.. code-block:: python

    trainer = Trainer(accelerator="hpu", devices=8, strategy="hpu_parallel", num_nodes=2)

In addition to this, the following environment variables need to be set to establish communication across nodes. Check out the documentation on :doc:`Cluster Environment <../clouds/cluster>` for more details.

- *MASTER_PORT* - required; has to be a free port on machine with NODE_RANK 0
- *MASTER_ADDR* - required (except for NODE_RANK 0); address of NODE_RANK 0 node
- *WORLD_SIZE* - required; how many workers are in the cluster
- *NODE_RANK* - required; id of the node in the cluster

The trainer needs to be instantiated on every node participating in the training.

On Node 1:

.. code-block:: bash

    MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> NODE_RANK=0 WORLD_SIZE=16
        python -m some_model_trainer.py (--arg1 ... train script args...)

On Node 2:

.. code-block:: bash

    MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> NODE_RANK=1 WORLD_SIZE=16
        python -m some_model_trainer.py (--arg1 ... train script args...)

----

How to access HPUs
------------------

To use HPUs, you must have access to a system with HPU devices.

AWS
^^^
You can either use `Gaudi-based AWS EC2 DL1 instances <https://aws.amazon.com/ec2/instance-types/dl1/>`__ or `Supermicro X12 Gaudi server <https://www.supermicro.com/en/solutions/habana-gaudi>`__ to get access to HPUs.

Check out the `PyTorch Model on AWS DL1 Instance Quick Start <https://docs.habana.ai/en/latest/AWS_EC2_DL1_and_PyTorch_Quick_Start/AWS_EC2_DL1_and_PyTorch_Quick_Start.html>`__.

----

.. _known-limitations_hpu:

Known limitations
-----------------

* `Habana dataloader <https://docs.habana.ai/en/latest/PyTorch_User_Guide/PyTorch_User_Guide.html#habana-data-loader>`__ is not supported.
