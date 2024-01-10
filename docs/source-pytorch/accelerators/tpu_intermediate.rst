:orphan:

TPU training (Intermediate)
===========================
**Audience:** Users looking to use cloud TPUs.

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

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

    import lightning as L

    my_model = MyLightningModule()
    trainer = L.Trainer(accelerator="tpu", devices=8)
    trainer.fit(my_model)

That's it! Your model will train on all 8 TPU cores.

----------------

16 bit precision
----------------
Lightning also supports training in 16-bit precision with TPUs.
By default, TPU training will use 32-bit precision. To enable it, do

.. code-block:: python

    import lightning as L

    my_model = MyLightningModule()
    trainer = L.Trainer(accelerator="tpu", precision="16-true")
    trainer.fit(my_model)

Under the hood the xla library will use the `bfloat16 type <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_.
