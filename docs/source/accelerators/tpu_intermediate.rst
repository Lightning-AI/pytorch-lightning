:orphan:

TPU training (Intermediate)
===========================
**Audience:** Users looking to use cloud TPUs.

----

DistributedSamplers
-------------------
Lightning automatically inserts the correct samplers - no need to do this yourself!

Usually, with TPUs (and DDP), you would need to define a DistributedSampler to move the right
chunk of data to the appropriate TPU. As mentioned, this is not needed in Lightning

.. note:: Don't add distributedSamplers. Lightning does this automatically

If for some reason you still need to, this is how to construct the sampler
for TPU use

.. code-block:: python

    import torch_xla.core.xla_model as xm


    def train_dataloader(self):
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

        # required for TPU support
        sampler = None
        if use_tpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
            )

        loader = DataLoader(dataset, sampler=sampler, batch_size=32)

        return loader

Configure the number of TPU cores in the trainer. You can only choose 1 or 8.
To use a full TPU pod skip to the TPU pod section.

.. code-block:: python

    import pytorch_lightning as pl

    my_model = MyLightningModule()
    trainer = pl.Trainer(accelerator="tpu", devices=8)
    trainer.fit(my_model)

That's it! Your model will train on all 8 TPU cores.

----------------

Distributed Backend with TPU
----------------------------
The ``accelerator`` option used for GPUs does not apply to TPUs.
TPUs work in DDP mode by default (distributing over each core)

----------------

TPU VM
------
Lightning supports training on the new Cloud TPU VMs.
Previously, we needed separate VMs to connect to the TPU machines, but as
Cloud TPU VMs run on the TPU Host machines, it allows direct SSH access
for the users. Hence, this architecture upgrade leads to cheaper and significantly
better performance and usability while working with TPUs.

The TPUVMs come pre-installed with latest versions of PyTorch and PyTorch XLA.
After connecting to the VM and before running your Lightning code, you would need
to set the XRT TPU device configuration.

.. code-block:: bash

    $ export XRT_TPU_CONFIG="localservice;0;localhost:51011"

You could learn more about the Cloud TPU VM architecture `here <https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_vms_3>`_

----------------

TPU Pod
-------
To train on more than 8 cores, your code actually doesn't change!
All you need to do is submit the following command:

.. code-block:: bash

    $ python -m torch_xla.distributed.xla_dist
    --tpu=$TPU_POD_NAME
    --conda-env=torch-xla-nightly
    -- python /usr/share/torch-xla-1.8.1/pytorch/xla/test/test_train_imagenet.py --fake_data

See `this guide <https://cloud.google.com/tpu/docs/tutorials/pytorch-pod>`_
on how to set up the instance groups and VMs needed to run TPU Pods.

----------------

16 bit precision
----------------
Lightning also supports training in 16-bit precision with TPUs.
By default, TPU training will use 32-bit precision. To enable 16-bit,
set the 16-bit flag.

.. code-block:: python

    import pytorch_lightning as pl

    my_model = MyLightningModule()
    trainer = pl.Trainer(accelerator="tpu", devices=8, precision=16)
    trainer.fit(my_model)

Under the hood the xla library will use the `bfloat16 type <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_.
