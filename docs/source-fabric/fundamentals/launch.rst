###########################
Launch distributed training
###########################

To run your code distributed across many devices and many machines, you need to do two things:

1. Configure Fabric with the number of devices and number of machines you want to use
2. Launch your code in multiple processes


----


*************
Simple Launch
*************

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric/animations/launch.mp4
    :width: 800
    :autoplay:
    :loop:
    :muted:
    :nocontrols:

You can configure and launch processes on your machine directly with Fabric's :meth:`~lightning.fabric.fabric.Fabric.launch` method:

.. code-block:: python

    # train.py
    ...

    # Configure accelerator, devices, num_nodes, etc.
    fabric = Fabric(devices=4, ...)

    # This launches itself into multiple processes
    fabric.launch()


In the command line, you run this like any other Python script:

.. code-block:: bash

    python train.py


This is the recommended way for running on a single machine and is the most convenient method for development and debugging.

It is also possible to use Fabric in a Jupyter notebook (including Google Colab, Kaggle, etc.) and launch multiple processes there.
You can learn more about it :ref:`here <Fabric in Notebooks>`.


----


*******************
Launch with the CLI
*******************

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric/animations/launch-cli.mp4
    :width: 800
    :autoplay:
    :loop:
    :muted:
    :nocontrols:

An alternative way to launch your Python script in multiple processes is to use the dedicated command line interface (CLI):

.. code-block:: bash

    fabric run path/to/your/script.py

This is essentially the same as running ``python path/to/your/script.py``, but it also lets you configure the following settings externally without changing your code:

- ``--accelerator``: The accelerator to use
- ``--devices``: The number of devices to use (per machine)
- ``--num_nodes``: The number of machines (nodes) to use
- ``--precision``: Which type of precision to use
- ``--strategy``: The strategy (communication layer between processes)


.. code-block:: bash

    fabric run --help

    Usage: fabric run [OPTIONS] SCRIPT [SCRIPT_ARGS]...

      Run a Lightning Fabric script.

      SCRIPT is the path to the Python script with the code to run. The script
      must contain a Fabric object.

      SCRIPT_ARGS are the remaining arguments that you can pass to the script
      itself and are expected to be parsed there.

    Options:
      --accelerator [cpu|gpu|cuda|mps|tpu]
                                      The hardware accelerator to run on.
      --strategy [ddp|dp|deepspeed]   Strategy for how to run across multiple
                                      devices.
      --devices TEXT                  Number of devices to run on (``int``), which
                                      devices to run on (``list`` or ``str``), or
                                      ``'auto'``. The value applies per node.
      --num-nodes, --num_nodes INTEGER
                                      Number of machines (nodes) for distributed
                                      execution.
      --node-rank, --node_rank INTEGER
                                      The index of the machine (node) this command
                                      gets started on. Must be a number in the
                                      range 0, ..., num_nodes - 1.
      --main-address, --main_address TEXT
                                      The hostname or IP address of the main
                                      machine (usually the one with node_rank =
                                      0).
      --main-port, --main_port INTEGER
                                      The main port to connect to the main
                                      machine.
      --precision [16-mixed|bf16-mixed|32-true|64-true|64|32|16|bf16]
                                      Double precision (``64-true`` or ``64``),
                                      full precision (``32-true`` or ``64``), half
                                      precision (``16-mixed`` or ``16``) or
                                      bfloat16 precision (``bf16-mixed`` or
                                      ``bf16``)
      --help                          Show this message and exit.



Here is how you run DDP with 8 GPUs and `torch.bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_ precision:

.. code-block:: bash

    fabric run ./path/to/train.py \
        --strategy=ddp \
        --devices=8 \
        --accelerator=cuda \
        --precision="bf16"

Or `DeepSpeed Zero3 <https://www.deepspeed.ai/2021/03/07/zero3-offload.html>`_ with mixed precision:

.. code-block:: bash

     fabric run ./path/to/train.py \
        --strategy=deepspeed_stage_3 \
        --devices=8 \
        --accelerator=cuda \
        --precision=16

:class:`~lightning.fabric.fabric.Fabric` can also figure it out automatically for you!

.. code-block:: bash

    fabric run ./path/to/train.py \
        --devices=auto \
        --accelerator=auto \
        --precision=16


----


.. _Fabric Cluster:

*******************
Launch on a Cluster
*******************

Fabric enables distributed training across multiple machines in several ways.
Choose from the following options based on your expertise level and available infrastructure.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Run single or multi-node on Lightning Studios
    :description: The easiest way to scale models in the cloud. No infrastructure setup required.
    :col_css: col-md-4
    :button_link: ../guide/multi_node/cloud.html
    :height: 160
    :tag: basic

.. displayitem::
    :header: SLURM Managed Cluster
    :description: Most popular for academic and private enterprise clusters.
    :col_css: col-md-4
    :button_link: ../guide/multi_node/slurm.html
    :height: 160
    :tag: intermediate

.. displayitem::
    :header: Bare Bones Cluster
    :description: Train across machines on a network using `torchrun`.
    :col_css: col-md-4
    :button_link: ../guide/multi_node/barebones.html
    :height: 160
    :tag: advanced

.. displayitem::
    :header: Other Cluster Environments
    :description: MPI, LSF, Kubeflow
    :col_css: col-md-4
    :button_link: ../guide/multi_node/other.html
    :height: 160
    :tag: advanced

.. raw:: html

        </div>
    </div>


----


**********
Next steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Mixed Precision Training
    :description: Save memory and speed up training using mixed precision
    :col_css: col-md-4
    :button_link: ../fundamentals/precision.html
    :height: 160
    :tag: basic

.. displayitem::
    :header: Distributed Communication
    :description: Learn all about communication primitives for distributed operation. Gather, reduce, broadcast, etc.
    :button_link: ../advanced/distributed_communication.html
    :col_css: col-md-4
    :height: 160
    :tag: advanced

.. raw:: html

        </div>
    </div>
