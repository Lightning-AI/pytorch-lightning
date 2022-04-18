########################################
Run on an on-prem cluster (intermediate)
########################################

.. _torch_distributed_run:

*************************
Run with TorchDistributed
*************************
`Torch Distributed Run <https://pytorch.org/docs/stable/elastic/run.html>`__ provides helper functions to setup distributed environment variables from the `PyTorch distributed communication package <https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization>`__ that need to be defined on each node.

Once the script is setup like described in :ref:` Training Script Setup<training_script_setup>`, you can run the below command across your nodes to start multi-node training.

Like a custom cluster, you have to ensure that there is network connectivity between the nodes with firewall rules that allow traffic flow on a specified *MASTER_PORT*.

Finally, you'll need to decide which node you'd like to be the main node (*MASTER_ADDR*), and the ranks of each node (*NODE_RANK*).

For example:

* *MASTER_ADDR* 10.10.10.16
* *MASTER_PORT* 29500
* *NODE_RANK* 0 for the first node, 1 for the second node

Run the below command with the appropriate variables set on each node.

.. code-block:: bash

    python -m torch.distributed.run
        --nnodes=2 # number of nodes you'd like to run with
        --master_addr <MASTER_ADDR>
        --master_port <MASTER_PORT>
        --node_rank <NODE_RANK>
        train.py (--arg1 ... train script args...)

.. note::

    ``torch.distributed.run`` assumes that you'd like to spawn a process per GPU if GPU devices are found on the node. This can be adjusted with ``-nproc_per_node``.

----

********
Get help
********
Setting up a cluster for distributed training is not trivial. Lightning offers lightning-grid which allows you to configure a cluster easily and run experiments via the CLI and web UI.

Try it out for free today:

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Train models on the cloud
   :description: Learn to run a model in the background on a cloud machine.
   :col_css: col-md-6
   :button_link: cloud_training.html
   :height: 150
   :tag: intermediate

.. raw:: html

        </div>
    </div
