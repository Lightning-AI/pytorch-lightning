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

Please refer to the :doc:`troubleshooting guide <../troubleshooting>` guide if you are experiencing issues related to multi-node training hanging or crashing.
If you are sick of troubleshooting cluster problems, give :doc:`Lightning Studios <./cloud>` a try!
For other questions, please don't hesitate to join the `Discord <https://discord.gg/VptPCZkGNa>`_.
