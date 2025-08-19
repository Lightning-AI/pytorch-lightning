########################################
Run on an on-prem cluster (intermediate)
########################################

.. _torch_distributed_run:

********************************
Run with TorchRun (TorchElastic)
********************************

`TorchRun <https://pytorch.org/docs/stable/elastic/run.html>`__ (previously known as TorchElastic) provides helper functions to set up distributed environment variables from the `PyTorch distributed communication package <https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization>`__ that need to be defined on each node.
Once the script is set up like described in :ref:`Training Script Setup <training_script_setup>`, you can run the below command across your nodes to start multi-node training.
Like a custom cluster, you have to ensure that there is network connectivity between the nodes with firewall rules that allow traffic flow on a specified *MASTER_PORT*.
Finally, you'll need to decide which node you'd like to be the main node (*MASTER_ADDR*), and the ranks of each node (*NODE_RANK*).

For example:

* **MASTER_ADDR:** 10.10.10.16
* **MASTER_PORT:** 29500
* **NODE_RANK:** 0 for the first node, 1 for the second node, etc.

Run the below command with the appropriate variables set on each node.

.. code-block:: bash

    torchrun \
        --nproc_per_node=<GPUS_PER_NODE> \
        --nnodes=<NUM_NODES> \
        --node_rank <NODE_RANK> \
        --master_addr <MASTER_ADDR> \
        --master_port <MASTER_PORT> \
        train.py --arg1 --arg2


- **--nproc_per_node:** Number of processes that will be launched per node (default 1). This number must match the number set in ``Trainer(devices=...)`` if specified in Trainer.
- **--nnodes:** Number of nodes/machines (default 1). This number must match the number set in ``Trainer(num_nodes=...)`` if specified in Trainer.
- **--node_rank:** The index of the node/machine.
- **--master_addr:** The IP address of the main node with node rank 0.
- **--master_port:** The port that will be used for communication between the nodes. Must be open in the firewall on each node to permit TCP traffic.

For more advanced configuration options in TorchRun such as elastic, fault-tolerant training, see the `official documentation <https://pytorch.org/docs/stable/elastic/run.html>`_.

|

**Example running on 2 nodes with 8 GPUs each:**

Assume the main node has the IP address 10.10.10.16.
On node the first node, you would run this command:

.. code-block:: bash

    torchrun \
        --nproc_per_node=8 --nnodes=2 --node_rank 0 \
        --master_addr 10.10.10.16 --master_port 50000 \
        train.py

On the second node, you would run this command:

.. code-block:: bash

    torchrun \
        --nproc_per_node=8 --nnodes=2 --node_rank 1 \
        --master_addr 10.10.10.16 --master_port 50000 \
        train.py

Note that the only difference between the two commands is the node rank!
