:orphan:

##################
Bare Bones Cluster
##################

**Audience**: Users who want to train on multiple machines that aren't part of a managed cluster.

This guide shows how to run a training job on a general-purpose cluster.
It assumes that you can log in to each machine and run commands.

Don't want to maintain your own infrastructure? Try the :doc:`Lightning cloud <./cloud>` instead.


----


************
Requirements
************

To set up a multi-node computing cluster, you need the following:

1. Multiple computers with Lightning installed
2. A network connectivity between the machines with firewall rules that allow traffic flow on a specified port.

|

We highly recommend setting up a shared filesystem to avoid the cumbersome copying of files between machines.


----


***************************
Prepare the training script
***************************

.. code-block:: python
    :caption: train.py

    from lightning.fabric import Fabric

    fabric = Fabric()

    # The rest of the training script
    ...

We intentionally omit to specify ``strategy``, ``devices``, and ``num_nodes`` here because these settings will get supplied through the CLI in the later steps.
You can still hard-code other options if you like.


----


*********************************
Launch the script on your cluster
*********************************

**Step 1**: Upload the training script and all needed files to the cluster.
Each node needs access to the same files.
If the nodes don't attach to a shared network drive, you'll need to upload the files to each node separately.

**Step 2**: Pick one of the nodes as your main node and write down its IP address.
Example: 10.10.10.16

**Step 3**: Launch the script on each node using the Lightning CLI.

In this example, we want to launch training across two nodes, each with 8 GPUs.
Log in to the **first node** and run this command:

.. code-block:: bash
    :emphasize-lines: 2,3

    fabric run \
        --node-rank=0  \
        --main-address=10.10.10.16 \
        --accelerator=cuda \
        --devices=8 \
        --num-nodes=2 \
        train.py

Log in to the **second node** and run this command:

.. code-block:: bash
    :emphasize-lines: 2,3

    fabric run \
        --node-rank=1  \
        --main-address=10.10.10.16 \
        --accelerator=cuda \
        --devices=8 \
        --num-nodes=2 \
        train.py

Note: The only difference between the two commands is the ``--node-rank`` setting, which identifies each node.
After executing these commands, you should immediately see an output like this:

.. code-block::

    Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/16
    Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/16
    ...


----


***************
Troubleshooting
***************


**My program is stuck initializing at startup. What is causing this?**

You are seeing a message like this in the logs, but nothing happens:

.. code-block::

    Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4

The most likely reasons and how to fix it:

- **Wrong network interface:** Some servers have multiple network interfaces.
  There is usually only one that can send and receive traffic from the network of the other nodes, but sometimes it is not set as the default.
  In this case, you need to set it manually:

  .. code-block:: bash

    export GLOO_SOCKET_IFNAME=eno1
    export NCCL_SOCKET_IFNAME=eno1
    fabric run ...

  You can find the interface name by parsing the output of the ``ifconfig`` command.
  The name of this interface **may differ on each node**.

- **NCCL can't communicate between the nodes:**

  Follow the steps in the `NCCL troubleshooting guide <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html>`_.
  In particular, take note of the network section that describes restricting the port range and firewall rules.

  .. code-block:: bash

      echo "net.ipv4.ip_local_port_range = 50000 51000" >> /etc/sysctl.conf
      sysctl --system
      ufw allow 50000:51000/tcp


**My program crashes with an NCCL error, but it is not helpful**

Launch your command by prepending ``NCCL_DEBUG=INFO`` to get more info.

.. code-block:: bash

    NCCL_DEBUG=INFO fabric run ...


----

If you are sick of troubleshooting cluster problems, give :doc:`Lightning cloud <./cloud>` a try!
For other questions, please don't hesitate to join the `Discord <https://discord.gg/VptPCZkGNa>`_.
