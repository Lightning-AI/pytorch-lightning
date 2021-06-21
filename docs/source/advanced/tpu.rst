.. _tpu:

TPU support
===========

.. raw:: html

    <video width="50%" max-width="400px" controls autoplay
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_tpus.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/tpu_cores.mp4"></video>

|

Lightning supports running on TPUs. At this moment, TPUs are available
on Google Cloud (GCP), Google Colab and Kaggle Environments. For more information on TPUs
`watch this video <https://www.youtube.com/watch?v=kPMpmcl_Pyw>`_.

----------------

TPU Terminology
---------------
A TPU is a Tensor processing unit. Each TPU has 8 cores where each
core is optimized for 128x128 matrix multiplies. In general, a single
TPU is about as fast as 5 V100 GPUs!

A TPU pod hosts many TPUs on it. Currently, TPU pod v2 has 2048 cores!
You can request a full pod from Google cloud or a "slice" which gives you
some subset of those 2048 cores.

----------------

How to access TPUs
------------------
To access TPUs, there are three main ways.

1. Using Google Colab.
2. Using Google Cloud (GCP).
3. Using Kaggle.

----------------

Kaggle TPUs
-----------
For starting Kaggle projects with TPUs, refer to this `kernel <https://www.kaggle.com/pytorchlightning/pytorch-on-tpu-with-pytorch-lightning>`_.

---------

Colab TPUs
----------
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

        !pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8-cp37-cp37m-linux_x86_64.whl

5. Once the above is done, install PyTorch Lightning (v 0.7.0+).

   .. code-block::

        !pip install pytorch-lightning

6. Then set up your LightningModule as normal.

----------------

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

----------------

TPU core training
-----------------

Lightning supports training on a single TPU core or 8 TPU cores.

The Trainer parameters ``tpu_cores`` defines how many TPU cores to train on (1 or 8) / Single TPU to train on [1].

For Single TPU training, Just pass the TPU core ID [1-8] in a list.

Single TPU core training. Model will train on TPU core ID 5.

.. code-block:: python

    trainer = pl.Trainer(tpu_cores=[5])

8 TPU cores training. Model will train on 8 TPU cores.

.. code-block:: python

    trainer = pl.Trainer(tpu_cores=8)

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
    trainer = pl.Trainer(tpu_cores=8, precision=16)
    trainer.fit(my_model)

Under the hood the xla library will use the `bfloat16 type <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_.


-----------------

Weight Sharing/Tying
--------------------
Weight Tying/Sharing is a technique where in the module weights are shared among two or more layers.
This is a common method to reduce memory consumption and is utilized in many State of the Art
architectures today.

PyTorch XLA requires these weights to be tied/shared after moving the model
to the TPU device. To support this requirement Lightning provides a model hook which is
called after the model is moved to the device. Any weights that require to be tied should
be done in the `on_post_move_to_device` model hook. This will ensure that the weights
among the modules are shared and not copied.

PyTorch Lightning has an inbuilt check which verifies that the model parameter lengths
match once the model is moved to the device. If the lengths do not match Lightning
throws a warning message.

Example:

.. code-block:: python

    from pytorch_lightning.core.lightning import LightningModule
    from torch import nn
    from pytorch_lightning.trainer.trainer import Trainer


    class WeightSharingModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(32, 10, bias=False)
            self.layer_2 = nn.Linear(10, 32, bias=False)
            self.layer_3 = nn.Linear(32, 10, bias=False)
            # TPU shared weights are copied independently
            # on the XLA device and this line won't have any effect.
            # However, it works fine for CPU and GPU.
            self.layer_3.weight = self.layer_1.weight

        def forward(self, x):
            x = self.layer_1(x)
            x = self.layer_2(x)
            x = self.layer_3(x)
            return x

        def on_post_move_to_device(self):
            # Weights shared after the model has been moved to TPU Device
            self.layer_3.weight = self.layer_1.weight


    model = WeightSharingModule()
    trainer = Trainer(max_epochs=1, tpu_cores=8)

See `XLA Documentation <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#xla-tensor-quirks>`_

-----------------------

Performance considerations
--------------------------

The TPU was designed for specific workloads and operations to carry out large volumes of matrix multiplication,
convolution operations and other commonly used ops in applied deep learning.
The specialization makes it a strong choice for NLP tasks, sequential convolutional networks, and under low precision operation.
There are cases in which training on TPUs is slower when compared with GPUs, for possible reasons listed:

- Too small batch size.
- Explicit evaluation of tensors during training, e.g. ``tensor.item()``
- Tensor shapes (e.g. model inputs) change often during training.
- Limited resources when using TPU's with PyTorch `Link <https://github.com/pytorch/xla/issues/2054#issuecomment-627367729>`_
- XLA Graph compilation during the initial steps `Reference <https://github.com/pytorch/xla/issues/2383#issuecomment-666519998>`_
- Some tensor ops are not fully supported on TPU, or not supported at all. These operations will be performed on CPU (context switch).
- PyTorch integration is still experimental. Some performance bottlenecks may simply be the result of unfinished implementation.

The official PyTorch XLA `performance guide <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#known-performance-caveats>`_
has more detailed information on how PyTorch code can be optimized for TPU. In particular, the
`metrics report <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#get-a-metrics-report>`_ allows
one to identify operations that lead to context switching.


About XLA
----------
XLA is the library that interfaces PyTorch with the TPUs.
For more information check out `XLA <https://github.com/pytorch/xla>`_.

Guide for `troubleshooting XLA <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md>`_
