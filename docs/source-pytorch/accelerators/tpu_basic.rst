:orphan:

TPU training (Basic)
====================
**Audience:** Users looking to train on single or multiple TPU cores.

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

----

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/tpu_cores.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_tpus.png
    :width: 400
    :muted:

Lightning supports running on TPUs. At this moment, TPUs are available
on Google Cloud (GCP), Google Colab and Kaggle Environments. For more information on TPUs
`watch this video <https://www.youtube.com/watch?v=kPMpmcl_Pyw>`_.

----------------

What is a TPU?
--------------
Tensor Processing Unit (TPU) is an AI accelerator application-specific integrated circuit (ASIC) developed by Google specifically for neural networks.

A TPU has 8 cores where each core is optimized for 128x128 matrix multiplies. In general, a single TPU is about as fast as 5 V100 GPUs!

A TPU pod hosts many TPUs on it. Currently, TPU v3 Pod has up to 2048 TPU cores and 32 TiB of memory!
You can request a full pod from Google cloud or a "slice" which gives you
some subset of those 2048 cores.

----

Run on TPU cores
----------------

To run on different cores, modify the ``devices`` argument.

.. code-block:: python

    # run on as many TPUs as available by default
    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto")
    # equivalent to
    trainer = Trainer()

    # run on one TPU core
    trainer = Trainer(accelerator="tpu", devices=1)
    # run on multiple TPU cores
    trainer = Trainer(accelerator="tpu", devices=8)
    # run on one specific TPU core: the 2nd core (index 1)
    trainer = Trainer(accelerator="tpu", devices=[1])
    # choose the number of cores automatically
    trainer = Trainer(accelerator="tpu", devices="auto")

----

How to access TPUs
------------------
To access TPUs, there are three main ways.

Google Colab
^^^^^^^^^^^^
Colab is like a jupyter notebook with a free GPU or TPU
hosted on GCP.

To get a TPU on colab, follow these steps:

1. Go to `Google Colab <https://colab.research.google.com/>`_.

2. Click "new notebook" (bottom right of pop-up).

3. Click runtime > change runtime settings. Select Python 3, and hardware accelerator "TPU".
   This will give you a TPU with 8 cores.

4. Next, insert this code into the first cell and execute.
   This will install the xla library that interfaces between PyTorch and the TPU.

   .. code-block::

        !pip install cloud-tpu-client https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.13-cp38-cp38m-linux_x86_64.whl

5. Once the above is done, install PyTorch Lightning.

   .. code-block::

        !pip install lightning

6. Then set up your LightningModule as normal.

Google Cloud (GCP)
^^^^^^^^^^^^^^^^^^
You could refer to this `page <https://cloud.google.com/tpu/docs/v4-users-guide>`_ for getting started with Cloud TPU resources on GCP.

----

Optimize Performance
--------------------

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

The official PyTorch XLA `performance guide <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#known-performance-caveats>`_
has more detailed information on how PyTorch code can be optimized for TPU. In particular, the
`metrics report <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#get-a-metrics-report>`_ allows
one to identify operations that lead to context switching.
