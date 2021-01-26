############
Accelerators
############
Accelerators connect a Lightning Trainer to arbitrary accelerators (CPUs, GPUs, TPUs, etc). Accelerators
also manage distributed accelerators (like DP, DDP, HPC cluster).

Accelerators can also be configured to run on arbitrary clusters using Plugins or to link up to arbitrary
computational strategies like 16-bit precision via AMP and Apex.

----------

******************************
Implement a custom accelerator
******************************
To link up arbitrary hardware, implement your own Accelerator subclass

.. code-block:: python

    from pytorch_lightning.accelerators.accelerator import Accelerator

        class MyAccelerator(Accelerator):
            def __init__(self, trainer, cluster_environment=None):
                super().__init__(trainer, cluster_environment)
                self.nickname = 'my_accelerator'

            def setup(self):
                # find local rank, etc, custom things to implement

            def train(self):
                # implement what happens during training

            def training_step(self):
                # implement how to do a training_step on this accelerator

            def validation_step(self):
                # implement how to do a validation_step on this accelerator

            def test_step(self):
                # implement how to do a test_step on this accelerator

            def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
                # implement how to do a backward pass with this accelerator

            def barrier(self, name: Optional[str] = None):
                # implement this accelerator's barrier

            def broadcast(self, obj, src=0):
                # implement this accelerator's broadcast function

            def sync_tensor(self,
                            tensor: Union[torch.Tensor],
                            group: Optional[Any] = None,
                            reduce_op: Optional[Union[ReduceOp, str]] = None) -> torch.Tensor:
                # implement how to sync tensors when reducing metrics across accelerators

********
Examples
********
The following examples illustrate customizing accelerators.

Example 1: Arbitrary HPC cluster
================================
To link any accelerator with an arbitrary cluster (SLURM, Condor, etc), pass in a Cluster Plugin which will be passed
into any accelerator.

First, implement your own ClusterEnvironment. Here is the torch elastic implementation.

.. code-block:: python

    import os
    from pytorch_lightning import _logger as log
    from pytorch_lightning.utilities import rank_zero_warn
    from pytorch_lightning.cluster_environments.cluster_environment import ClusterEnvironment

    class TorchElasticEnvironment(ClusterEnvironment):

        def __init__(self):
            super().__init__()

        def master_address(self):
            if "MASTER_ADDR" not in os.environ:
                rank_zero_warn(
                    "MASTER_ADDR environment variable is not defined. Set as localhost"
                )
                os.environ["MASTER_ADDR"] = "127.0.0.1"
            log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
            master_address = os.environ.get('MASTER_ADDR')
            return master_address

        def master_port(self):
            if "MASTER_PORT" not in os.environ:
                rank_zero_warn(
                    "MASTER_PORT environment variable is not defined. Set as 12910"
                )
                os.environ["MASTER_PORT"] = "12910"
            log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

            port = os.environ.get('MASTER_PORT')
            return port

        def world_size(self):
            return os.environ.get('WORLD_SIZE')

        def local_rank(self):
            return int(os.environ['LOCAL_RANK'])

Now, pass it into the trainer which will use Torch Elastic across your accelerator of choice.

.. code-block:: python

    cluster = TorchElasticEnvironment()
    accelerator = MyAccelerator()
    trainer = Trainer(plugins=[cluster], accelerator=MyAccelerator())

In this example, MyAccelerator can define arbitrary hardware (like IPUs or TPUs) and links it to an arbitrary
compute cluster.

------------

**********************
Available Accelerators
**********************

CPU Accelerator
===============

.. autoclass:: pytorch_lightning.accelerators.cpu_accelerator.CPUAccelerator
    :noindex:

DDP Accelerator
===============

.. autoclass:: pytorch_lightning.accelerators.ddp_accelerator.DDPAccelerator
    :noindex:

DDP2 Accelerator
================

.. autoclass:: pytorch_lightning.accelerators.ddp2_accelerator.DDP2Accelerator
    :noindex:

DDP CPU HPC Accelerator
=======================

.. autoclass:: pytorch_lightning.accelerators.ddp_cpu_hpc_accelerator.DDPCPUHPCAccelerator
    :noindex:

DDP CPU Spawn Accelerator
=========================

.. autoclass:: pytorch_lightning.accelerators.ddp_cpu_spawn_accelerator.DDPCPUSpawnAccelerator
    :noindex:

DDP HPC Accelerator
===================

.. autoclass:: pytorch_lightning.accelerators.ddp_hpc_accelerator.DDPHPCAccelerator
    :noindex:

DDP Spawn Accelerator
=====================

.. autoclass:: pytorch_lightning.accelerators.ddp_spawn_accelerator.DDPSpawnAccelerator
    :noindex:

GPU Accelerator
===============

.. autoclass:: pytorch_lightning.accelerators.gpu_accelerator.GPUAccelerator
    :noindex:

Horovod Accelerator
===================

.. autoclass:: pytorch_lightning.accelerators.horovod_accelerator.HorovodAccelerator
    :noindex:

TPU Accelerator
===============

.. autoclass:: pytorch_lightning.accelerators.tpu_accelerator.TPUAccelerator
    :noindex:
