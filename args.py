import os
import sys
import utils
import argparse
import numpy as np

def get_default_layer_args(arglist):
    """ layer / block specific parameters. Default values are given here """

    parser = argparse.ArgumentParser(description='Vector Quantization')
    add = parser.add_argument

    # Quantization settings
    add('--num_codebooks', type=int, default=1,
            help='Number of codebooks')
    add('--embed_dim', type=int, default=100,
            help='Embedding size, `D` in paper')
    add('--num_embeddings', type=int, default=128,
            help='Number of embeddings to choose from, `K` in paper')
    add('--commitment_cost', type=float, default=1,
            help='Beta in the loss function')
    add('--decay', type=float, default=0.99,
            help='Moving av decay for codebook update')
    add('--quant_size', type=int, nargs='+', default=[1], help='Height of Tensor Quantization')
    add('--model', type=str, choices=['vqvae', 'gumbel', 'argmax'], default='vqvae')
    add('--learning_rate', type=float, default=1e-3)

    add('--embed_grad_update', type=int, default=1)

    # VQVAE model, defaults like in paper
    add('--num_hiddens', type=int, default=100,
            help="Number of channels for Convolutions, not ResNet")
    add('--num_residual_hiddens', type=int, default = 100,
            help="Number of channels for ResNet")
    add('--num_residual_layers', type=int, default=1)
    add('--stride', type=int, nargs='+', default=[2], help='use if strides are uneven across H/W')
    add('--downsample', type=int, default=1, help='downsampling at every layer')

    return parser.parse_args(arglist)


def get_global_args(arglist):
    """ Regular (not layer specific) arguments """

    parser = argparse.ArgumentParser(description='Vector Quantization')
    add = parser.add_argument

    # Data and training settings
    add('--data_folder', type=str, default="../cl-pytorch/data",
            help='Location of data (will download data if does not exist)')
    add('--dataset', type=str, default='split_cifar10',
            help='Dataset name')
    add('--data_size', type=int, nargs='+', default=(3, 128, 128),
            help='height / width of the input. Note that only Imagenet supports this')
    add('--batch_size', type=int, default=10)
    add('--max_iterations', type=int, default=None,
            help="Max it per epoch, for debugging (default: None)")
    add('--num_epochs', type=int, default=1,
            help='number of epochs (default: 40)')
    add('--device', type=str, default='cuda')

    # CL specific
    add('--override_cl_defaults', action='store_true')
    add('--n_classes_per_task', type=int, default=-1)
    add('--multiple_heads', type=int, default=-1)

    # new ones
    add('--global_learning_rate', type=float, default=1e-4)
    add('--optimization', type=str, default='blockwise', choices=['blockwise', 'global'])
    add('--name', type=str, default='basic')

    add('--num_blocks', type=int, default=0, help='number of QLayers in QStack')

    add('--xyz', action='store_true', help='if True, xyz coordinates are used instead of polar')
    add('--from_compressed', type=int, default=1)

    # Misc
    add('--seed', type=int, default=521)
    add('--debug', action='store_true')
    add('--recon_th', type=float, nargs='+', default=[1e-3], help='satisfying reconstruction threshold')

    add('--gen_weights', type=str, default=None)

    # From old repo
    add('--max_task', type=int, default=-1)
    add('--n_iters', type=int, default=1)
    add('--samples_per_task', type=int, default=-1)
    add('--update_representations', type=int, default=1)
    add('--rehearsal', type=int, default=1)
    add('--mem_size', type=int, default=600)
    add('--n_classes', type=int, default=100)
    add('--n_runs', type=int, default=1)
    add('--n_epochs', type=int, default=1)

    add('--print_logs', type=int, default=1)
    add('--sunk_cost', action='store_true', help='if true, we do not substract model weights')

    # ablation
    add('--no_idx_update', action='store_true')

    # classifier args
    add('--cls_lr', type=float, default=0.1)
    add('--cls_n_iters', type=int, default=1)
    add('--test_on_recon', action='store_true')

    args = parser.parse_args(arglist)

    return args


def get_args():
    # Assumption 1: specific block parameters are separated by three dashes
    # Assumption 2: specific block parameters are specified AFTER regular args
    # e.g. python main.py --batch_size 32 --num_blocks 2 ---block_1 --num_hiddens 32 ---block_2

    layer_flags = [i for i in range(len(sys.argv)) if sys.argv[i].startswith('---layer')]

    global_args = sys.argv[1:len(sys.argv) if len(layer_flags) == 0 else min(layer_flags)]
    global_args = get_global_args(global_args)

    global_args.layers = {}

    # now we add layer specific params
    for i in range(len(layer_flags)):
        layer_idx   = layer_flags[i]
        end_idx     = len(sys.argv) if (i+1) == len(layer_flags) else layer_flags[i+1]
        layer_no    = int(sys.argv[layer_idx].split('---layer_')[-1])
        layer_args  = get_default_layer_args(sys.argv[layer_idx + 1:end_idx])

        ''' (for now) copy the remaining args manually '''
        layer_args.optimization = global_args.optimization
        layer_args.rehearsal    = global_args.rehearsal
        layer_args.mem_size     = global_args.mem_size
        layer_args.n_classes    = global_args.n_classes
        layer_args.data_size    = global_args.data_size

        # make sure layer does not exist yet
        assert layer_no not in global_args.layers.keys()
        global_args.layers[layer_no] = layer_args

    # for now let's specify every layer via the command line
    assert len(layer_flags) == global_args.num_blocks    or global_args.gen_weights
    assert len(layer_flags) == len(global_args.recon_th) or global_args.gen_weights

    assert global_args.cls_n_iters <= global_args.n_iters, 'might as well train gen. model more?'

    # we want to know what the compression factor is at every level
    current_shape = global_args.data_size[1:] # e.g. (128, 128)

    fill = lambda x : (str(x) + (20 - len(str(x))) * ' ')[:20]
    print(fill('INPUT SHAPE'), fill('LATENT SHAPE'), fill('COMP RATE'), 'ARGMIN SHAPE')

    # specify remaining args
    for i in range(global_args.num_blocks):

        # original code had `i` instead of `i+1` for `global_args` index (I think the latter is correct)
        input_channels = global_args.data_size[0] if i == 0 else global_args.layers[i - 1].embed_dim
        global_args.layers[i].in_channel = global_args.layers[i].out_channel = input_channels


        ''' stride '''
        stride = global_args.layers[i].stride
        if len(stride) == 1:
            stride = stride * 2
            global_args.layers[i].stride = stride


        ''' compression rate '''
        comp_map = {1:1, 2:1, 4:2}
        per_dim_ds = comp_map[global_args.layers[i].downsample]

        input_shape   = current_shape
        current_shape = (current_shape[0] // (stride[0] ** per_dim_ds),
                         current_shape[1] // (stride[1] ** per_dim_ds))

        # parse the quantization sizes:
        total_idx = 0.
        argmin_shapes = []

        for cb_idx in range(len(global_args.layers[i].quant_size)):
            qs = global_args.layers[i].quant_size[cb_idx]
            if qs > 100: # for now we encode quant size (4, 2) as 402
                qH, qW = qs // 100, qs % 100
                qs = (qH, qW)

            ''' quant size '''
            if type(qs) == int:
                qs = (qs, qs)

            global_args.layers[i].quant_size[cb_idx] = qs

            # count the amount of indices for a specific block
            total_idx += np.prod(current_shape) / np.prod(qs)

            argmin_shapes += [(current_shape[0] // qs[0], current_shape[1] // qs[1])]

        global_args.layers[i].comp_rate = np.prod(global_args.data_size) / float(total_idx)
        global_args.layers[i].comp_rate *= np.log2(256) / np.log2(global_args.layers[i].num_embeddings)
        global_args.layers[i].argmin_shapes = argmin_shapes

        len_stride = len(global_args.layers[i].stride)
        assert len_stride <= 2
        if len_stride == 1:
            global_args.layers[i].stride = global_args.layers[i].stride *  2

        print(fill(input_shape), fill(current_shape), fill(global_args.layers[i].comp_rate), argmin_shapes)

        # the rest is simply renaming
        global_args.layers[i].channel    = global_args.layers[i].num_hiddens


    args = global_args
    if args.gen_weights:
        model_id = args.gen_weights.split('_')[-1]
        args.model_name = 'loaded_model_{}'.format(model_id)
    else:
        args.model_name = 'DS{}_NB{}_EGU{}_Comp{}_Coef{:.2f}_{}'.format(
                                args.dataset[-10:],
                                args.num_blocks,
                                args.layers[0].embed_grad_update,
                                ''.join(['{:.2f}^'.format(args.layers[i].comp_rate) for i in range(args.num_blocks)]),
                                args.layers[0].decay + args.layers[0].commitment_cost,
                                np.random.randint(10000))
        args.model_name = 'test' if args.debug else args.model_name

    return args


def get_debug_args():
    sys.argv[1:] = ['--num_blocks', '2', '---layer_0', '--num_embeddings', '100', '--embed_dim', '44', '---layer_1', '--model', 'argmax', '--num_codebooks', '2', '--quant_size', '2', '1', '--num_embeddings', '100', '--embed_dim', '44']
    return get_args()
