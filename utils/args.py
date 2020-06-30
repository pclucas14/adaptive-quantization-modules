import os
import sys
import utils
import argparse
import numpy as np


def get_global_args():
    """ Regular (not layer specific) arguments """

    parser = argparse.ArgumentParser(description='Vector Quantization')
    add = parser.add_argument

    # Data and training settings
    add('--data_folder', type=str, default="../cl-pytorch/data",
            help='Location of data (will download data if does not exist)')
    add('--run_dir', type=str, default='runs',
            help='base directory in which all experiment logs will be held')
    add('--dataset', type=str, default='split_cifar10',
            choices=['split_cifar10','split_cifar100','miniimagenet',
                'processed_kitti'],
            help='Dataset name')
    add('--data_size', type=int, nargs='+', default=(3, 128, 128),
            help='height / width of the input. Note that only Imagenet'        +
            ' supports the use of this flag')
    add('--batch_size', type=int, default=10,
            help='batch size of the incoming data stream')
    add('--buffer_batch_size', type=int, default=10,
            help='batch size used for rehearsal. Setting it to `-1` will make' +
            'buffer_batch_size equal to batch_size')
    add('--num_epochs', type=int, default=1,
            help='number of epochs per task. Use 1 for online learning')
    add('--device', type=str, default='cuda')
    add('--name', type=str, default='test')


    # CL specific
    add('--max_task', type=int, default=-1,
            help='maximum amount of tasks. Use `-1` for no maximum')
    add('--override_cl_defaults', action='store_true',
            help='use this flag if you want to change the number of classes '  +
            'pert task, or whether to run single-headed or multi-headed')
    add('--n_classes_per_task', type=int, default=-1,
            help='number of classes per task. Will only be used if '           +
            '`--override_cl_defaults`')
    add('--multiple_heads', type=int, default=-1,
            help='Use `1` for multi-head experiments and `0` for single head. '+
            'Will only be used if `--override_cl_defaults`')


    # Misc
    add('--seed', type=int, default=521)
    add('--debug', action='store_true')


    # From old repo
    add('--n_iters', type=int, default=1,
            help='number of iterations to perform on incoming data')
    add('--samples_per_task', type=int, default=-1,
            help='number of samples per CL task. Use `-1` for all samples')
    add('--rehearsal', type=int, default=1,
            help='whether to rehearse on previous data samples from the buffer')
    add('--mem_size', type=int, default=600,
            help='size of memory allowed. Measured in number of real examples '+
            'stored. If mem_size == 500, then 500 * np.prod(data_size) floats '+
            'will be the size of the memory')
    add('--n_classes', type=int, default=100,
            help='number of classes in dataset')
    add('--n_runs', type=int, default=1,
            help='number of runs for a specific configuration')

    # classifier args
    add('--cls_lr', type=float, default=0.05,
            help='learning rate for the classifier')
    add('--cls_n_iters', type=int, default=1,
            help='number of iterations on the incoming data for the classifier')

    add('--config', type=str, default='config/cifar_20.yaml')
    add('--gen_weights', type=str, default=None)

    add('--mode', type=str, default='offline', choices=['online', 'offline'])

    return parser.parse_args()


def get_args():
    return get_global_args()
