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
from docs.source.examples.example_model import ExampleModel
# ---------------------

AVAILABLE_MODELS = {
    'model_template': ExampleModel
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
    exp = Experiment(
        name='test_tube_exp',
        debug=True,
        save_dir=log_dir,
        version=0,
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
        save_function=None,
        save_best_only=True,
        verbose=True,
        monitor=hparams.model_save_monitor_value,
        mode=hparams.model_save_monitor_mode
    )

    # gpus are ; separated for inside a node and , within nodes
    gpu_list = None
    if hparams.gpus is not None:
        gpu_list = [int(x) for x in hparams.gpus.split(';')]

    # configure trainer
    trainer = Trainer(
        experiment=exp,
        cluster=cluster,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        gpus=gpu_list
    )

    # train model
    trainer.fit(model)


def get_default_parser(strategy, root_dir):

    possible_model_names = list(AVAILABLE_MODELS.keys())
    parser = HyperOptArgumentParser(strategy=strategy, add_help=False)
    add_default_args(parser, root_dir, possible_model_names=possible_model_names, rand_seed=SEED)
    return parser


def get_model_name(args):
    for i, arg in enumerate(args):
        if 'model_name' in arg:
            return args[i+1]


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
    cluster.job_time = '48:00:00'
    cluster.gpu_type = '1080ti'
    cluster.memory_mb_per_node = 48000

    # any modules for code to run in env
    cluster.add_command('source activate pytorch_lightning')

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

    model_name = get_model_name(sys.argv)
    if model_name is None:
        model_name = 'model_template'

    # use default args
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = get_default_parser(strategy='random_search', root_dir=root_dir)

    # allow model to overwrite or extend args
    TRAINING_MODEL = AVAILABLE_MODELS[model_name]
    parser = TRAINING_MODEL.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # format GPU layout
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # ---------------------
    # RUN TRAINING
    # ---------------------

    # cluster and CPU
    if hyperparams.on_cluster:
        # run on HPC cluster
        print('RUNNING ON SLURM CLUSTER')
        gpu_ids = hyperparams.gpus.split(';')
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids)
        optimize_on_cluster(hyperparams)

    elif hyperparams.gpus is None:
        # run on cpu
        print('RUNNING ON CPU')
        main(hyperparams, None, None)

    # single or multiple GPUs on same machine
    gpu_ids = hyperparams.gpus.split(';')
    if hyperparams.interactive:
        # run on 1 gpu
        print(f'RUNNING INTERACTIVE MODE ON GPUS. gpu ids: {gpu_ids}')
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids)
        main(hyperparams, None, None)

    else:
        # multiple GPUs on same machine
        print(f'RUNNING MULTI GPU. GPU ids: {gpu_ids}')
        hyperparams.optimize_parallel_gpu(
            main_local,
            gpu_ids=gpu_ids,
            nb_trials=hyperparams.nb_hopt_trials,
            nb_workers=len(gpu_ids)
        )
