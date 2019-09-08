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
    add('--embed_dim', type=int, default=64,
            help='Embedding size, `D` in paper')
    add('--num_embeddings', type=int, default=256,
            help='Number of embeddings to choose from, `K` in paper')
    add('--commitment_cost', type=float, default=0.25,
            help='Beta in the loss function')
    add('--decay', type=float, default=0.99,
            help='Moving av decay for codebook update')
    add('--quant_size', type=int, nargs='+', default=[1], help='Height of Tensor Quantization')
    add('--model', type=str, choices=['vqvae', 'gumbel', 'argmax'], default='vqvae')
    add('--learning_rate', type=float, default=1e-3)

    # VQVAE model, defaults like in paper
    add('--enc_height', type=int, default=8,
            help="Encoder output size, used for downsampling and KL")
    add('--num_hiddens', type=int, default=128,
            help="Number of channels for Convolutions, not ResNet")
    add('--num_residual_hiddens', type=int, default = 32,
            help="Number of channels for ResNet")
    add('--num_residual_layers', type=int, default=2)
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
    add('--data_size', type=int, default=128,
            help='height / width of the input. Note that only Imagenet supports this')
    add('--batch_size', type=int, default=10)
    add('--max_iterations', type=int, default=None,
            help="Max it per epoch, for debugging (default: None)")
    add('--num_epochs', type=int, default=1,
            help='number of epochs (default: 40)')
    add('--device', type=str, default='cuda')

    # new ones
    add('--global_learning_rate', type=float, default=1e-4)
    add('--optimization', type=str, default='blockwise', choices=['blockwise', 'global'])
    add('--num_blocks', type=int, default=1, help='number of QLayers in QStack')
    add('--input_channels', type=int, default=3, help='3 for RBG, 2 for Lidar')
    add('--max_task', type=int, default=-1)

    add('--xyz', action='store_true', help='if True, xyz coordinates are used instead of polar')

    # Misc
    add('--seed', type=int, default=521)
    add('--debug', action='store_true')

    # From old repo
    add('--n_iters', type=int, default=1)
    add('--samples_per_task', type=int, default=-1)
    add('--update_representations', type=int, default=1)
    add('--rehearsal', type=int, default=1)
    add('--mem_size', type=int, default=600)

    args = parser.parse_args(arglist)

    return args


def get_args():
    # Assumption 1: specific block parameters are separated by three dashes
    # Assumption 2: specific block parameters are specified AFTER regular args
    # e.g. python main.py --batch_size 32 --num_blocks 2 ---block_1 --num_hiddens 32 ---block_2 ---enc_height 64

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

        # make sure layer does not exist yet
        assert layer_no not in global_args.layers.keys()
        global_args.layers[layer_no] = layer_args

    # for now let's specify every layer via the command line
    assert len(layer_flags) == global_args.num_blocks

    # specify remaining args
    for i in range(global_args.num_blocks):
        input_size = global_args.data_size if i == 0 else global_args.layers[i-1].enc_height

        # original code had `i` instead of `i+1` for `global_args` index (I think the latter is correct)
        input_channels = global_args.input_channels if i == 0 else global_args.layers[i - 1].embed_dim
        global_args.layers[i].in_channel = global_args.layers[i].out_channel = input_channels

        # parse the quantization sizes:
        for cb_idx in range(len(global_args.layers[i].quant_size)):
            qs = global_args.layers[i].quant_size[cb_idx]
            if qs > 100: # for now we encode quant size (4, 2) as 402
                qH, qW = qs // 100, qs % 100
                global_args.layers[i].quant_size[cb_idx] = (qH, qW)

        len_stride = len(global_args.layers[i].stride)
        assert len_stride <= 2
        if len_stride == 1:
            global_args.layers[i].stride = global_args.layers[i].stride *  2

        # the rest is simply renaming
        global_args.layers[i].channel    = global_args.layers[i].num_hiddens


    args = global_args
    args.model_name = 'M:{}_DS:{}_NB:{}_NI:{}_OPT:{}_UR:{}_Re:{}_{}'.format(args.layers[0].model[:5], args.dataset[:10],
                                                 args.num_blocks, args.n_iters,
                                                 args.optimization[:5], args.update_representations, args.rehearsal,
                                                 np.random.randint(10000))
    args.model_name = 'test' if args.debug else args.model_name

    return args


def get_debug_args():
    sys.argv[1:] = ['--num_blocks', '2', '---layer_0', '--enc_height', '64', '--num_embeddings', '100', '--embed_dim', '44', '---layer_1', '--model', 'argmax', '--enc_height', '32', '--num_codebooks', '2', '--quant_size', '2', '1', '--num_embeddings', '100', '--embed_dim', '44']
    return get_args()
