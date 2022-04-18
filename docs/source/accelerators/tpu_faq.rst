:orphan:

TPU training (Basic)
====================
**Audience:** Users looking to train on single or multiple TPU cores.

----

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_tpus.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/tpu_cores.mp4"></video>

|

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

Run on 1 TPU core
-----------------
Enable the following Trainer arguments to run on 1 TPU.

.. code::

    trainer = Trainer(accelerator="tpu", devices=1)

----

Run on multiple TPU cores
-------------------------
For multiple TPU cores, change the value of the devices flag.

.. code::

    trainer = Trainer(accelerator="tpu", devices=8)

----

Run on a specific TPU core
--------------------------

To run on a specific core, specify the index of the TPU core.

.. code-block:: python

    trainer = pl.Trainer(accelerator="tpu", devices=[5])

This example runs on the 5th core, not on five cores.

----

How to access TPUs
------------------
To access TPUs, there are three main ways.

Google Colab
^^^^^^^^^^^^
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

        !pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

5. Once the above is done, install PyTorch Lightning.

   .. code-block::

        !pip install pytorch-lightning

6. Then set up your LightningModule as normal.

Google Cloud (GCP)
^^^^^^^^^^^^^^^^^^
?

Kaggle
^^^^^^
For starting Kaggle projects with TPUs, refer to this `kernel <https://www.kaggle.com/pytorchlightning/pytorch-on-tpu-with-pytorch-lightning>`_.

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
- PyTorch integration is still experimental. Some performance bottlenecks may simply be the result of unfinished implementation.

The official PyTorch XLA `performance guide <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#known-performance-caveats>`_
has more detailed information on how PyTorch code can be optimized for TPU. In particular, the
`metrics report <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#get-a-metrics-report>`_ allows
one to identify operations that lead to context switching.

----

FAQ
---

**XLA configuration is missing**

.. code-block::

    File "/usr/local/lib/python3.8/dist-packages/torch_xla/core/xla_model.py", line 18, in <lambda>
        _DEVICES = xu.LazyProperty(lambda: torch_xla._XLAC._xla_get_devices())
    RuntimeError: tensorflow/compiler/xla/xla_client/computation_client.cc:273 : Missing XLA configuration
    Traceback (most recent call last):
    ...
    File "/home/kaushikbokka/pytorch-lightning/pytorch_lightning/utilities/device_parser.py", line 125, in parse_tpu_cores
        raise MisconfigurationException('No TPU devices were found.')
    pytorch_lightning.utilities.exceptions.MisconfigurationException: No TPU devices were found.

This means the system is missing XLA configuration. You would need to set up XRT TPU device configuration.

For TPUVM architecture, you could set it in your terminal by:

.. code-block:: bash

    export XRT_TPU_CONFIG="localservice;0;localhost:51011"

And for the old TPU + 2VM architecture, you could set it by:

.. code-block:: bash

    export TPU_IP_ADDRESS=10.39.209.42  # You could get the IP Address in the GCP TPUs section
    export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

----

**How to clear up the programs using TPUs in the background**

.. code-block:: bash

    lsof -w /lib/libtpu.so | grep "python" |  awk '{print $2}' | xargs -r kill -9

Sometimes, there can still be old programs running on the TPUs, which would make the TPUs unavailable to use. You could use the above command in the terminal to kill the running processes.

----

**Replication issue**

.. code-block::

    File "/usr/local/lib/python3.6/dist-packages/torch_xla/core/xla_model.py", line 200, in set_replication
        replication_devices = xla_replication_devices(devices)
    File "/usr/local/lib/python3.6/dist-packages/torch_xla/core/xla_model.py", line 187, in xla_replication_devices
        .format(len(local_devices), len(kind_devices)))
    RuntimeError: Cannot replicate if number of devices (1) is different from 8

This error is raised when the XLA device is called outside the spawn process. Internally in `TPUSpawn` Strategy for training on multiple tpu cores, we use XLA's `xmp.spawn`.
Don't use ``xm.xla_device()`` while working on Lightning + TPUs!

----

**Unsupported datatype transfer to TPU**

.. code-block::

    File "/usr/local/lib/python3.8/dist-packages/torch_xla/utils/utils.py", line 205, in _for_each_instance_rewrite
        v = _for_each_instance_rewrite(result.__dict__[k], select_fn, fn, rwmap)
    File "/usr/local/lib/python3.8/dist-packages/torch_xla/utils/utils.py", line 206, in _for_each_instance_rewrite
        result.__dict__[k] = v
    TypeError: 'mappingproxy' object does not support item assignment

PyTorch XLA only supports Tensor objects for CPU to TPU data transfer. Might cause issues if the User is trying to send some non-tensor objects through the DataLoader or during saving states.

----

**Using `tpu_spawn_debug` Strategy alias**

.. code-block:: python

    import pytorch_lightning as pl

    my_model = MyLightningModule()
    trainer = pl.Trainer(accelerator="tpu", devices=8, strategy="tpu_spawn_debug")
    trainer.fit(my_model)

Example Metrics report:

.. code-block::

    Metric: CompileTime
        TotalSamples: 202
        Counter: 06m09s401ms746.001us
        ValueRate: 778ms572.062us / second
        Rate: 0.425201 / second
        Percentiles: 1%=001ms32.778us; 5%=001ms61.283us; 10%=001ms79.236us; 20%=001ms110.973us; 50%=001ms228.773us; 80%=001ms339.183us; 90%=001ms434.305us; 95%=002ms921.063us; 99%=21s102ms853.173us


A lot of PyTorch operations aren't lowered to XLA, which could lead to significant slowdown of the training process.
These operations are moved to the CPU memory and evaluated, and then the results are transferred back to the XLA device(s).
By using the `tpu_spawn_debug` Strategy, users could create a metrics report to diagnose issues.

The report includes things like (`XLA Reference <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#troubleshooting>`_):

* how many times we issue XLA compilations and time spent on issuing.
* how many times we execute and time spent on execution
* how many device data handles we create/destroy etc.

----

**TPU Pod Training Startup script**

All TPU VMs in a Pod setup are required to access the model code and data.
One easy way to achieve this is to use the following startup script when creating the TPU VM pod.
It will perform the data downloading on all TPU VMs. Note that you need to export the corresponding environment variables following the instruction in Create TPU Node.

.. code-block:: bash

    gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID} --accelerator-type v3-32 --version ${RUNTIME_VERSION} --metadata startup-script=setup.py

Then users could ssh to any TPU worker, e.g. worker 0, check if data/model downloading is finished and
start the training after generating the ssh-keys to ssh between VM workers on a pod:

.. code-block:: bash

    python3 -m torch_xla.distributed.xla_dist --tpu=$TPU_NAME -- python3 train.py --max_epochs=5 --batch_size=32
