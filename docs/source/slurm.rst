.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer

Computing cluster (SLURM)
=========================

Lightning automates job the details behind  training on a SLURM powered cluster.

.. _multi-node:

Multi-node training
-------------------
To train a model using multiple-nodes do the following:

1.  Design your LightningModule.

2.  Enable ddp in the trainer

    .. code-block:: python

       # train on 32 GPUs across 4 nodes
       trainer = Trainer(gpus=8, num_nodes=4, distributed_backend='ddp')

3.  It's a good idea to structure your train.py file like this:

    .. testcode::

        # train.py
        def main(hparams):
            model = LightningTemplateModel(hparams)

            trainer = pl.Trainer(
                gpus=8,
                num_nodes=4,
                distributed_backend='ddp'
            )

            trainer.fit(model)


        if __name__ == '__main__':
            root_dir = os.path.dirname(os.path.realpath(__file__))
            parent_parser = ArgumentParser(add_help=False)
            hyperparams = parser.parse_args()

            # TRAIN
            main(hyperparams)

4.  Create the appropriate SLURM job

    .. code-block:: bash

        # (submit.sh)
        #!/bin/bash -l

        # SLURM SUBMIT SCRIPT
        #SBATCH --nodes=4
        #SBATCH --gres=gpu:8
        #SBATCH --ntasks-per-node=8
        #SBATCH --mem=0
        #SBATCH --time=0-02:00:00

        # activate conda env
        source activate $1

        # -------------------------
        # debugging flags (optional)
         export NCCL_DEBUG=INFO
         export PYTHONFAULTHANDLER=1

        # on your cluster you might need these:
        # set the network interface
        # export NCCL_SOCKET_IFNAME=^docker0,lo

        # might need the latest cuda
        # module load NCCL/2.4.7-1-cuda.10.0
        # -------------------------

        # run script from above
        srun python3 train.py

5.  If you want auto-resubmit (read below), add this line to the submit.sh script

    .. code-block:: bash

        #SBATCH --signal=SIGUSR1@90

6.  Submit the SLURM job

    .. code-block:: bash

        sbatch submit.sh

.. note:: using :class:`~torch.utils.data.distributed.DistributedSampler` is already handled by Lightning.

Walltime auto-resubmit
----------------------
When you use Lightning in a SLURM cluster, lightning automatically detects when it is about
to run into the walltime, and it does the following:

1.  Saves a temporary checkpoint.
2.  Requeues the job.
3.  When the job starts, it loads the temporary checkpoint.

To get this behavior make sure to add the correct signal to your SLURM script

.. code-block::

    # 90 seconds before training ends
    #SBATCH --signal=SIGUSR1@90
