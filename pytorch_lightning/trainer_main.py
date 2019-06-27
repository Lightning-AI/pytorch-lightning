import os
import sys

import torch
import numpy as np
from test_tube import HyperOptArgumentParser, Experiment, SlurmCluster
from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning.utils.arg_parse import add_default_args
from time import sleep

from pytorch_lightning.callbacks.pt_callbacks import EarlyStopping, ModelCheckpoint
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------
# DEFINE MODEL HERE
# ---------------------
from pytorch_lightning.models.sample_model_template.model_template import ExampleModel1
# ---------------------

AVAILABLE_MODELS = {
    'model_1': ExampleModel1
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
    on_gpu = torch.cuda.is_available()
    if hparams.disable_cuda:
        on_gpu = False

    device = 'cuda' if on_gpu else 'cpu'
    hparams.__setattr__('device', device)
    hparams.__setattr__('on_gpu', on_gpu)
    hparams.__setattr__('nb_gpus', torch.cuda.device_count())
    hparams.__setattr__('inference_mode', hparams.model_load_weights_path is not None)

    # delay each training start to not overwrite logs
    process_position, current_gpu = TRAINING_MODEL.get_process_position(hparams.gpus)
    sleep(process_position + 1)

    # init experiment
    exp = Experiment(
        name=hparams.tt_name,
        debug=hparams.debug,
        save_dir=hparams.tt_save_path,
        version=hparams.hpc_exp_number,
        autosave=False,
        description=hparams.tt_description
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

    # configure trainer
    trainer = Trainer(
        experiment=exp,
        on_gpu=on_gpu,
        cluster=cluster,
        progress_bar=hparams.enable_tqdm,
        overfit_pct=hparams.overfit,
        track_grad_norm=hparams.track_grad_norm,
        fast_dev_run=hparams.fast_dev_run,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        process_position=process_position,
        current_gpu_name=current_gpu,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        enable_early_stop=hparams.enable_early_stop,
        max_nb_epochs=hparams.max_nb_epochs,
        min_nb_epochs=hparams.min_nb_epochs,
        train_percent_check=hparams.train_percent_check,
        val_percent_check=hparams.val_percent_check,
        test_percent_check=hparams.test_percent_check,
        val_check_interval=hparams.val_check_interval,
        log_save_interval=hparams.log_save_interval,
        add_log_row_interval=hparams.add_log_row_interval,
        lr_scheduler_milestones=hparams.lr_scheduler_milestones
    )

    # train model
    trainer.fit(model)


def get_default_parser(strategy, root_dir):

    possible_model_names = list(AVAILABLE_MODELS.keys())
    parser = HyperOptArgumentParser(strategy=strategy, add_help=False)
    add_default_args(parser, root_dir, possible_model_names, SEED)
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

    # use default args
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = get_default_parser(strategy='random_search', root_dir=root_dir)

    # allow model to overwrite or extend args
    TRAINING_MODEL = AVAILABLE_MODELS[model_name]
    parser = TRAINING_MODEL.add_model_specific_args(parent_parser)
    parser.json_config('-c', '--config', default=root_dir + '/run_configs/local.json')
    hyperparams = parser.parse_args()

    # format GPU layout
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpu_ids = hyperparams.gpus.split(';')

    # RUN TRAINING
    if hyperparams.on_cluster:
        print('RUNNING ON SLURM CLUSTER')
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids)
        optimize_on_cluster(hyperparams)

    elif hyperparams.single_run_gpu:
        print(f'RUNNING 1 TRIAL ON GPU. gpu: {gpu_ids[0]}')
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        main(hyperparams, None, None)

    elif hyperparams.local or hyperparams.single_run:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        print('RUNNING LOCALLY')
        main(hyperparams, None, None)

    else:
        print(f'RUNNING MULTI GPU. GPU ids: {gpu_ids}')
        hyperparams.optimize_parallel_gpu(
            main_local,
            gpu_ids=gpu_ids,
            nb_trials=hyperparams.nb_hopt_trials,
            nb_workers=len(gpu_ids)
        )
