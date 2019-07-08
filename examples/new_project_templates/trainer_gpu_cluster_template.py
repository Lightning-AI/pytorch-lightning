import os
import sys
import numpy as np
from time import sleep
import torch

from test_tube import HyperOptArgumentParser, Experiment, SlurmCluster
from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning.utils.arg_parse import add_default_args

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------
# DEFINE MODEL HERE
# ---------------------
from lightning_module_template import LightningTemplateModel
# ---------------------

AVAILABLE_MODELS = {
    'model_template': LightningTemplateModel
}


"""
Allows training by using command line arguments
Run by:
# TYPE YOUR RUN COMMAND HERE
"""


def main_local(hparams):
    main(hparams, None, None)


def main(hparams, cluster, results_dict):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    on_gpu = hparams.gpus is not None and torch.cuda.is_available()

    device = 'cuda' if on_gpu else 'cpu'
    hparams.__setattr__('device', device)
    hparams.__setattr__('on_gpu', on_gpu)
    hparams.__setattr__('nb_gpus', torch.cuda.device_count())
    hparams.__setattr__('inference_mode', hparams.model_load_weights_path is not None)

    # delay each training start to not overwrite logs
    process_position, current_gpu = TRAINING_MODEL.get_process_position(hparams.gpus)
    sleep(process_position + 1)

    # init experiment
    log_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(log_dir, 'test_tube_demo_logs')
    exp = Experiment(
        name='test_tube_exp',
        save_dir=log_dir,
        autosave=False,
        description='test demo'
    )

    exp.argparse(hparams)
    exp.save()

    # build model
    print('loading model...')
    model = TRAINING_MODEL(hparams)
    print('model built')

    # callbacks
    early_stop = EarlyStopping(
        monitor=hparams.early_stop_metric,
        patience=hparams.early_stop_patience,
        verbose=True,
        mode=hparams.early_stop_mode
    )

    model_save_path = '{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        verbose=True,
        monitor=hparams.model_save_monitor_value,
        mode=hparams.model_save_monitor_mode
    )

    # configure trainer
    trainer = Trainer(
        experiment=exp,
        cluster=cluster,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        gpus=hparams.gpus,
        nb_gpu_nodes=hyperparams.nb_gpu_nodes
    )

    # train model
    trainer.fit(model)


def get_default_parser(strategy, root_dir):

    possible_model_names = list(AVAILABLE_MODELS.keys())
    parser = HyperOptArgumentParser(strategy=strategy, add_help=False)
    add_default_args(parser, root_dir, possible_model_names=possible_model_names, rand_seed=SEED)
    return parser


def optimize_on_cluster(hyperparams):
    # enable cluster training
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.tt_save_path,
        test_tube_exp_name=hyperparams.tt_name
    )

    # email for cluster coms
    cluster.notify_job_status(email='add_email_here', on_done=True, on_fail=True)

    # configure cluster
    cluster.per_experiment_nb_gpus = hyperparams.per_experiment_nb_gpus
    cluster.per_experiment_nb_nodes = hyperparams.nb_gpu_nodes
    cluster.job_time = '2:00:00'
    cluster.gpu_type = 'volta'
    cluster.memory_mb_per_node = 0

    # any modules for code to run in env
    cluster.add_command('source activate lightning')
    cluster.add_slurm_cmd(cmd='constraint', value='volta32gb', comment='use 32gb gpus')
    cluster.add_slurm_cmd(cmd='partition', value=hyperparams.gpu_partition, comment='use 32gb gpus')

    # name of exp
    job_display_name = hyperparams.tt_name.split('_')[0]
    job_display_name = job_display_name[0:3]

    # run hopt
    print('submitting jobs...')
    cluster.optimize_parallel_cluster_gpu(
        main,
        nb_trials=hyperparams.nb_hopt_trials,
        job_name=job_display_name
    )


if __name__ == '__main__':

    # use default args
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = get_default_parser(strategy='random_search', root_dir=root_dir)

    # cluster args not defined inside the model
    parent_parser.add_argument('-gpu_partition', type=str)
    parent_parser.add_argument('-per_experiment_nb_gpus', type=int)
    parent_parser.add_argument('--nb_gpu_nodes', type=int, default=1)

    # allow model to overwrite or extend args
    parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    # run on HPC cluster
    print('RUNNING ON SLURM CLUSTER')
    optimize_on_cluster(hyperparams)
