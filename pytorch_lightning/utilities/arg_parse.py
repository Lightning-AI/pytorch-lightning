"""
List of default args which mught be useful for all the available flags
Might need to update with the new flags
"""

import os


def add_default_args(parser, root_dir, rand_seed=None, possible_model_names=None):
    # training, test, val check intervals
    parser.add_argument('--eval_test_set', dest='eval_test_set', action='store_true',
                        help='true = run test set also')
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int,
                        help='check val every n epochs')
    parser.opt_list('--accumulate_grad_batches', default=1, type=int, tunable=False,
                    help='accumulates gradients k times before applying update.'
                         ' Simulates huge batch size')
    parser.add_argument('--max_nb_epochs', default=200, type=int, help='cap epochs')
    parser.add_argument('--min_nb_epochs', default=2, type=int, help='min epochs')
    parser.add_argument('--train_percent_check', default=1.0, type=float,
                        help='how much of training set to check')
    parser.add_argument('--val_percent_check', default=1.0, type=float,
                        help='how much of val set to check')
    parser.add_argument('--test_percent_check', default=1.0, type=float,
                        help='how much of test set to check')

    parser.add_argument('--val_check_interval', default=0.95, type=float,
                        help='how much within 1 epoch to check val')
    parser.add_argument('--log_save_interval', default=100, type=int,
                        help='how many batches between log saves')
    parser.add_argument('--row_log_interval', default=100, type=int,
                        help='add log every k batches')

    # early stopping
    parser.add_argument('--disable_early_stop', dest='enable_early_stop', action='store_false')
    parser.add_argument('--early_stop_metric', default='val_acc', type=str)
    parser.add_argument('--early_stop_mode', default='min', type=str)
    parser.add_argument('--early_stop_patience', default=3, type=int,
                        help='number of epochs until stop')

    # gradient handling
    parser.add_argument('--gradient_clip_val', default=-1, type=int)
    parser.add_argument('--track_grad_norm', default=-1, type=int,
                        help='if > 0, will track this grad norm')

    # model saving
    parser.add_argument('--model_save_path', default=root_dir + '/model_weights')
    parser.add_argument('--model_save_monitor_value', default='val_acc')
    parser.add_argument('--model_save_monitor_mode', default='max')

    # model paths
    parser.add_argument('--model_load_weights_path', default=None, type=str)

    if possible_model_names is not None:
        parser.add_argument('--model_name', default='', help=','.join(possible_model_names))

    # test_tube settings
    parser.add_argument('-en', '--tt_name', default='pt_test')
    parser.add_argument('-td', '--tt_description', default='pytorch lightning test')
    parser.add_argument('--tt_save_path', default=os.path.join(root_dir, 'test_tube_logs'),
                        help='logging dir')
    parser.add_argument('--enable_single_run', dest='single_run', action='store_true')
    parser.add_argument('--nb_hopt_trials', default=1, type=int)
    parser.add_argument('--log_stdout', dest='log_stdout', action='store_true')

    # GPU
    parser.add_argument('--gpus', default=None, type=str)
    parser.add_argument('--single_run_gpu', dest='single_run_gpu', action='store_true')
    parser.add_argument('--default_tensor_type', default='torch.cuda.FloatTensor', type=str)
    parser.add_argument('--use_amp', dest='use_amp', action='store_true')
    parser.add_argument('--check_grad_nans', dest='check_grad_nans', action='store_true')
    parser.add_argument('--amp_level', default='O2', type=str)

    # run on hpc
    parser.add_argument('--on_cluster', dest='on_cluster', action='store_true')

    # FAST training
    # use these settings to make sure network has no bugs without running a full dataset
    parser.add_argument('--fast_dev_run', dest='fast_dev_run', default=False, action='store_true',
                        help='runs validation after 1 training step')
    parser.add_argument('--enable_tqdm', dest='enable_tqdm', default=False, action='store_true',
                        help='false removes the progress bar')
    parser.add_argument('--overfit', default=-1, type=float,
                        help='% of dataset to use with this option. float, or -1 for none')

    # debug args
    if rand_seed is not None:
        parser.add_argument('--random_seed', default=rand_seed, type=int)

    parser.add_argument('--interactive', dest='interactive', action='store_true',
                        help='runs on gpu without cluster')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='enables/disables test tube')
    parser.add_argument('--local', dest='local', action='store_true',
                        help='enables local training')

    # optimizer
    parser.add_argument('--lr_scheduler_milestones', default=None, type=str)
