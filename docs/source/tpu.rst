TPU support
===========

Lightning supports running on TPUs. At this moment, TPUs are available
on Google Cloud (GCP), Google Colab and Kaggle Environments. For more information on TPUs
`watch this video <https://www.youtube.com/watch?v=kPMpmcl_Pyw>`_.

---------------

Live demo
----------
Check out this `Google Colab <https://colab.research.google.com/drive/1-_LKx4HwAxl5M6xPJmqAAu444LTDQoa3>`_ to see how to train MNIST on TPUs.

---------------

TPU Terminology
---------------
A TPU is a Tensor processing unit. Each TPU has 8 cores where each
core is optimized for 128x128 matrix multiplies. In general, a single
TPU is about as fast as 5 V100 GPUs!

A TPU pod hosts many TPUs on it. Currently, TPU pod v2 has 2048 cores!
You can request a full pod from Google cloud or a "slice" which gives you
some subset of those 2048 cores.

---------------

How to access TPUs
-------------------
To access TPUs there are two main ways.

1. Using google colab.
2. Using Google Cloud (GCP).
3. Using Kaggle.

---------------

Colab TPUs
-----------
Colab is like a jupyter notebook with a free GPU or TPU
hosted on GCP.

To get a TPU on colab, follow these steps:

1. Go to `https://colab.research.google.com/ <https://colab.research.google.com/>`_.

2. Click "new notebook" (bottom right of pop-up).

3. Click runtime > change runtime settings. Select Python 3, and hardware accelerator "TPU".
   This will give you a TPU with 8 cores.

4. Next, insert this code into the first cell and execute.
   This will install the xla library that interfaces between PyTorch and the TPU.

   .. code-block::

    !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
    !python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev

5. Once the above is done, install PyTorch Lightning (v 0.7.0+).

   .. code-block::

        !pip install pytorch-lightning

6. Then set up your LightningModule as normal.

---------------

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
        dataset = MNIST(
            os.getcwd(),
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        # required for TPU support
        sampler = None
        if use_tpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True
            )

        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=32
        )

        return loader

Configure the number of TPU cores in the trainer. You can only choose 1 or 8.
To use a full TPU pod skip to the TPU pod section.

.. code-block:: python

    import pytorch_lightning as pl

    my_model = MyLightningModule()
    trainer = pl.Trainer(tpu_cores=8)
    trainer.fit(my_model)

That's it! Your model will train on all 8 TPU cores.

---------------

Single TPU core training
----------------------------
Lightning supports training on a single TPU core. Just pass the TPU core ID [1-8] in a list.

.. code-block:: python

    trainer = pl.Trainer(tpu_cores=[1])

---------------

Distributed Backend with TPU
----------------------------
The ```distributed_backend``` option used for GPUs does not apply to TPUs.
TPUs work in DDP mode by default (distributing over each core)

---------------

TPU Pod
--------
To train on more than 8 cores, your code actually doesn't change!
All you need to do is submit the following command:

.. code-block:: bash

    $ python -m torch_xla.distributed.xla_dist
    --tpu=$TPU_POD_NAME
    --conda-env=torch-xla-nightly
    -- python /usr/share/torch-xla-0.5/pytorch/xla/test/test_train_imagenet.py --fake_data

---------------

16 bit precision
-----------------
Lightning also supports training in 16-bit precision with TPUs.
By default, TPU training will use 32-bit precision. To enable 16-bit, also
set the 16-bit flag.

.. code-block:: python

    import pytorch_lightning as pl

    my_model = MyLightningModule()
    trainer = pl.Trainer(tpu_cores=8, precision=16)
    trainer.fit(my_model)

Under the hood the xla library will use the `bfloat16 type <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_.

---------------

About XLA
----------
XLA is the library that interfaces PyTorch with the TPUs.
For more information check out `XLA <https://github.com/pytorch/xla>`_.

Guide for `troubleshooting XLA <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md>`_
