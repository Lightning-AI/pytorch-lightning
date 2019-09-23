"""
Multi-node example (GPU)
"""
import os
import numpy as np
from time import sleep
import torch

from test_tube import HyperOptArgumentParser, Experiment, SlurmCluster
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from examples.new_project_templates.lightning_module_template import LightningTemplateModel

PORT = np.random.randint(12000, 20000, 1)[0]
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def main_local(hparams):
    main(hparams, None, None)


def main(hparams, cluster):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    print('loading model...')
    model = LightningTemplateModel(hparams)
    print('model built')

    # ------------------------
    # 2 INIT TEST TUBE EXP
    # ------------------------
    # when using grid search, it's possible for all models to start at once
    # and use the same test tube experiment version
    relative_node_id = int(os.environ['SLURM_NODEID'])
    sleep(relative_node_id + 1)

    # init experiment
    exp = Experiment(
        name=hyperparams.experiment_name,
        save_dir=hyperparams.test_tube_save_path,
        autosave=False,
        version=hparams.hpc_exp_number,  # match the slurm job version number
        description='test demo'
    )

    exp.argparse(hparams)
    exp.save()

    # ------------------------
    # 3 DEFINE CALLBACKS
    # ------------------------
    model_save_path = '{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)
    early_stop = EarlyStopping(
        monitor='val_acc',
        patience=3,
        verbose=True,
        mode='max'
    )

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        experiment=exp,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        gpus=hparams.per_experiment_nb_gpus,
        nb_gpu_nodes=hyperparams.nb_gpu_nodes
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)


def optimize_on_cluster(hyperparams):
    # enable cluster training
    # log all scripts to the test tube folder
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.slurm_log_path,
    )

    # email for cluster coms
    cluster.notify_job_status(email=hyperparams.email, on_done=True, on_fail=True)

    # configure cluster
    cluster.per_experiment_nb_gpus = hyperparams.per_experiment_nb_gpus
    cluster.per_experiment_nb_nodes = hyperparams.nb_gpu_nodes
    cluster.job_time = '2:00:00'
    cluster.gpu_type = 'volta'
    cluster.memory_mb_per_node = 0

    # any modules for code to run in env
    cluster.add_command(f'source activate {hyperparams.conda_env}')

    # set DDP master port
    cluster.add_command(f'export MASTER_PORT={PORT}')

    # OPTIONAL for debugging
    # without these flags errors in your code will 
    # appear to be nccl errors
    cluster.add_command('export NCCL_DEBUG=INFO')
    cluster.add_command('export PYTHONFAULTHANDLER=1')

    # depending on your cluster config, you probably want
    # to limit the wired connection device
    # cluster.add_command('export NCCL_SOCKET_IFNAME=^docker0,lo')

    # depending on your cluster, you might need to load
    # the latest NCCL version
    # cluster.load_modules(['NCCL/2.4.7-1-cuda.10.0'])

    # run only on 32GB voltas
    cluster.add_slurm_cmd(cmd='constraint', value='volta32gb',
                          comment='use 32gb gpus')
    cluster.add_slurm_cmd(cmd='partition', value=hyperparams.gpu_partition,
                          comment='use 32gb gpus')

    # run hopt
    # creates and submits jobs to slurm
    cluster.optimize_parallel_cluster_gpu(
        main,
        nb_trials=hyperparams.num_hyperparam_trials,
        job_name=hyperparams.experiment_name
    )


if __name__ == '__main__':

    # use default args
    root_dir = os.path.dirname(os.path.realpath(__file__))
    demo_log_dir = os.path.join(root_dir, 'pt_lightning_demo_logs')

    checkpoint_dir = os.path.join(demo_log_dir, 'model_weights')
    test_tube_dir = os.path.join(demo_log_dir, 'test_tube_data')
    slurm_out_dir = os.path.join(demo_log_dir, 'slurm_scripts')

    parent_parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)

    # cluster args not defined inside the model

    parent_parser.add_argument('--per_experiment_nb_gpus', type=int,
                               default=8, help='how many gpus to use in a node')
    parent_parser.add_argument('--nb_gpu_nodes', type=int, default=2,
                               help='how many nodes to use in a cluster')
    parent_parser.add_argument('--test_tube_save_path', type=str, default=test_tube_dir,
                               help='where to save logs')
    parent_parser.add_argument('--slurm_log_path', type=str, default=slurm_out_dir,
                               help='where to save slurm meta')
    parent_parser.add_argument('--model_save_path', type=str, default=checkpoint_dir,
                               help='where to save model')
    parent_parser.add_argument('--experiment_name', type=str, default='pt_lightning_exp_a',
                               help='test tube exp name')
    parent_parser.add_argument('--num_hyperparam_trials', type=int, default=6,
                               help='how many grid search trials to run')

    parent_parser.add_argument('--email', type=str, default='add@email.com',
                               help='email for jobs')
    parent_parser.add_argument('--conda_env', type=str, default='base',
                               help='email for jobs')
    parent_parser.add_argument('--gpu_partition', type=str, help='consult your cluster manual')

    # allow model to overwrite or extend args
    parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    # run on HPC cluster
    print('RUNNING ON SLURM CLUSTER')
    optimize_on_cluster(hyperparams)
