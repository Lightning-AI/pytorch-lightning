:orphan:

.. _tpu_faq:

TPU training (FAQ)
==================

**********************************************************
How to clear up the programs using TPUs in the background?
**********************************************************

.. code-block:: bash

    pgrep python |  awk '{print $2}' | xargs -r kill -9

Sometimes, there can still be old programs running on the TPUs, which would make the TPUs unavailable to use. You could use the above command in the terminal to kill the running processes.

----

*************************************
How to resolve the replication issue?
*************************************

.. code-block::

    File "/usr/local/lib/python3.6/dist-packages/torch_xla/core/xla_model.py", line 200, in set_replication
        replication_devices = xla_replication_devices(devices)
    File "/usr/local/lib/python3.6/dist-packages/torch_xla/core/xla_model.py", line 187, in xla_replication_devices
        .format(len(local_devices), len(kind_devices)))
    RuntimeError: Cannot replicate if number of devices (1) is different from 8

This error is raised when the XLA device is called outside the spawn process. Internally in the XLA-Strategy for training on multiple tpu cores, we use XLA's `xmp.spawn`.
Don't use ``xm.xla_device()`` while working on Lightning + TPUs!

----

**************************************
Unsupported datatype transfer to TPUs?
**************************************

.. code-block::

    File "/usr/local/lib/python3.8/dist-packages/torch_xla/utils/utils.py", line 205, in _for_each_instance_rewrite
        v = _for_each_instance_rewrite(result.__dict__[k], select_fn, fn, rwmap)
    File "/usr/local/lib/python3.8/dist-packages/torch_xla/utils/utils.py", line 206, in _for_each_instance_rewrite
        result.__dict__[k] = v
    TypeError: 'mappingproxy' object does not support item assignment

PyTorch XLA only supports Tensor objects for CPU to TPU data transfer. Might cause issues if the User is trying to send some non-tensor objects through the DataLoader or during saving states.

----

*************************************************
How to setup the debug mode for Training on TPUs?
*************************************************

.. code-block:: python

    import lightning as L

    my_model = MyLightningModule()
    trainer = L.Trainer(accelerator="tpu", devices=8, strategy="xla_debug")
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
By using the `xla_debug` Strategy, users could create a metrics report to diagnose issues.

The report includes things like (`XLA Reference <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#troubleshooting>`_):

* how many times we issue XLA compilations and time spent on issuing.
* how many times we execute and time spent on execution
* how many device data handles we create/destroy etc.
