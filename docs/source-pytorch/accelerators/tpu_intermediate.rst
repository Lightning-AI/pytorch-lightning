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

All TPU VMs in a Pod setup are required to access the model code and data.
One easy way to achieve this is to use the following startup script when creating the TPU VM pod.
It will perform the data downloading on all TPU VMs. Note that you need to export the corresponding environment variables following the instruction in Create TPU Node.

.. code-block:: bash

    gcloud alpha compute tpus tpu-vm create ${TPU_POD_NAME} --zone ${ZONE} --project ${PROJECT_ID} --accelerator-type ${ACCELERATOR_TYPE} --version ${RUNTIME_VERSION} --metadata startup-script=setup.py

Then you could ssh to any TPU worker, e.g. worker 0, check if data/model downloading is finished and
start the training after generating the ssh-keys to ssh between VM workers on a pod.
All you need to do is submit the following command:

.. code-block:: bash

    python3 -m torch_xla.distributed.xla_dist --tpu=$TPU_POD_NAME -- python3 train.py --max_epochs=5 --batch_size=32

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
