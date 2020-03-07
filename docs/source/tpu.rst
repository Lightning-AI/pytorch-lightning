TPU support
===========

Lightning supports running on TPUs. At this moment, TPUs are only available
on Google Cloud (GCP). For more information on TPUs
`watch this video <https://www.youtube.com/watch?v=kPMpmcl_Pyw>`_.

Live demo
----------
Check out this `Google Colab <https://colab.research.google.com/drive/1-_LKx4HwAxl5M6xPJmqAAu444LTDQoa3>`_ to see how to train MNIST on TPUs.

TPU Terminology
---------------
A TPU is a Tensor processing unit. Each TPU has 8 cores where each
core is optimized for 128x128 matrix multiplies. In general, a single
TPU is about as fast as 5 V100 GPUs!

A TPU pod hosts many TPUs on it. Currently, TPU pod v2 has 2048 cores!
You can request a full pod from Google cloud or a "slice" which gives you
some subset of those 2048 cores.

How to access TPUs
-------------------
To access TPUs there are two main ways.

1. Using google colab.
2. Using Google Cloud (GCP).

Colab TPUs
-----------
Colab is like a jupyter notebook with a free GPU or TPU
hosted on GCP.

To get a TPU on colab, follow these steps:

1. Go to https://colab.research.google.com/.

2. Click "new notebook" (bottom right of pop-up).

3. Click runtime > change runtime settings. Select Python 3,
and hardware accelerator "TPU". This will give you a TPU with 8 cores.

4. Next, insert this code into the first cell and execute. This
will install the xla library that interfaces between PyTorch and
the TPU.

.. code-block:: python

    import collections
    from datetime import datetime, timedelta
    import os
    import requests
    import threading

    _VersionConfig = collections.namedtuple('_VersionConfig', 'wheels,server')
    VERSION = "xrt==1.15.0"  #@param ["xrt==1.15.0", "torch_xla==nightly"]
    CONFIG = {
        'xrt==1.15.0': _VersionConfig('1.15', '1.15.0'),
        'torch_xla==nightly': _VersionConfig('nightly', 'XRT-dev{}'.format(
            (datetime.today() - timedelta(1)).strftime('%Y%m%d'))),
    }[VERSION]
    DIST_BUCKET = 'gs://tpu-pytorch/wheels'
    TORCH_WHEEL = 'torch-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
    TORCH_XLA_WHEEL = 'torch_xla-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
    TORCHVISION_WHEEL = 'torchvision-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)

    # Update TPU XRT version
    def update_server_xrt():
      print('Updating server-side XRT to {} ...'.format(CONFIG.server))
      url = 'http://{TPU_ADDRESS}:8475/requestversion/{XRT_VERSION}'.format(
          TPU_ADDRESS=os.environ['COLAB_TPU_ADDR'].split(':')[0],
          XRT_VERSION=CONFIG.server,
      )
      print('Done updating server-side XRT: {}'.format(requests.post(url)))

    update = threading.Thread(target=update_server_xrt)
    update.start()

    # Install Colab TPU compat PyTorch/TPU wheels and dependencies
    !pip uninstall -y torch torchvision
    !gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" .
    !gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" .
    !gsutil cp "$DIST_BUCKET/$TORCHVISION_WHEEL" .
    !pip install "$TORCH_WHEEL"
    !pip install "$TORCH_XLA_WHEEL"
    !pip install "$TORCHVISION_WHEEL"
    !sudo apt-get install libomp5
    update.join()
5. Once the above is done, install PyTorch Lightning (v 0.7.0+).

.. code-block::

    ! pip install pytorch-lightning

6. Then set up your LightningModule as normal.

7. TPUs require a DistributedSampler. That means you should change your
train_dataloader (and val, train) code as follows.

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

8. Configure the number of TPU cores in the trainer. You can only choose
1 or 8. To use a full TPU pod skip to the TPU pod section.

.. code-block:: python

    import pytorch_lightning as pl

    my_model = MyLightningModule()
    trainer = pl.Trainer(num_tpu_cores=8)
    trainer.fit(my_model)

That's it! Your model will train on all 8 TPU cores.

TPU Pod
--------
To train on more than 8 cores, your code actually doesn't change!
All you need to do is submit the following command:

.. code-block:: bash

    $ python -m torch_xla.distributed.xla_dist
    --tpu=$TPU_POD_NAME
    --conda-env=torch-xla-nightly
    -- python /usr/share/torch-xla-0.5/pytorch/xla/test/test_train_imagenet.py --fake_data

16 bit precision
-----------------
Lightning also supports training in 16-bit precision with TPUs.
By default, TPU training will use 32-bit precision. To enable 16-bit, also
set the 16-bit flag.

.. code-block:: python

    import pytorch_lightning as pl

    my_model = MyLightningModule()
    trainer = pl.Trainer(num_tpu_cores=8, precision=16)
    trainer.fit(my_model)

Under the hood the xla library will use the `bfloat16 type <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_.


About XLA
----------
XLA is the library that interfaces PyTorch with the TPUs.
For more information check out `XLA <https://github.com/pytorch/xla>`_.

Guide for `troubleshooting XLA <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md>`_
