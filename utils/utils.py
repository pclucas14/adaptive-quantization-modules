import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.nn.utils.weight_norm as wn
from collections import OrderedDict as OD
from collections import defaultdict as DD
from copy import deepcopy
from PIL import Image

import matplotlib
matplotlib.use('pdf')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# good'ol utils
# ---------------------------------------------------------------------------------

PRINT = ['recon', 'samples', 'decay', 'avg_l2', 'comp_rate', 'drift', 'capacity']

class RALog():
    """ keeps track of running averages of values """

    def __init__(self):
        self.reset()

    def reset(self):
        self.storage  = OD()
        self.count    = OD()
        self.per_task = OD()

    def one_liner(self):
        # import pdb; pdb.set_trace()
        fill = lambda x, y : (x + ' ' * max(0, y - len(x)))[-y:]
        out = ''
        for key, value in self.storage.items():
            if sum([x in key for x in PRINT]) > 0:
                out += fill(key, 20)
                if type(value) == np.ndarray:
                    out += str(value)
                else:
                    out += '{:.4f}\t'.format(value)

        return out

    def log(self, key, value, per_task=True):
        if 'tensor' in str(type(value)).lower():
            if value.numel() == 1:
                value = value.item()
            else:
                value = value.cpu().data.numpy()

        if key not in self.storage.keys():
            self.count[key] = 1
            self.storage[key] = value
            self.per_task[key] = per_task
        else:
            prev = self.storage[key]
            cnt  = self.count[key]
            self.storage[key] = (prev * cnt  + value) / (cnt + 1.)
            self.count[key] += 1


def average_log(dic):

    keys = dic.keys()
    # get unique keys
    keys = [x.split('_')[0] for x in keys]
    keys = list(set(keys))

    avgs = {}
    for super_key in keys:
        current = OD()
        for key in dic.keys():
            if super_key in key:
                task = int(key.split('_')[1])
                current[task] = sum([x for x in dic[key]]) / len(dic[key])
        avgs[super_key] = current

    return avgs


def make_histogram(values, title, tmp_path='tmp.png'):
    plt.clf()
    max_idx = np.argwhere(values > 0)

    if max_idx.shape[0] > 0:
        max_idx = max_idx.max()
    else:
        max_idx = values.shape[0]

    values = values[:max_idx]

    plt.bar(np.arange(values.shape[0]), values)

    plt.title(title)
    plt.grid(True)

    plt.savefig(tmp_path, quality=10)

    np_img = np.array(Image.open(tmp_path))[:, :, :3]
    return np_img.transpose(2, 0, 1)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def save_args(args, path):
    c_args = vars(deepcopy(args))

    # 1) convert all nested (1st level) dics to namespaces
    n_layers = len(args.layers)

    for layer_idx in range(n_layers):
        c_args['layers'][layer_idx] = vars(deepcopy(args.layers[layer_idx]))

    with open(os.path.join(path, 'args.json'), 'w') as f:
        json.dump(c_args, f)


def load_args(path):
    with open(os.path.join(path, 'args.json'), 'r') as f:
        args_dict = json.load(f)

    # 1) convert all nested (1st level) dics to namespaces
    n_layers = len(args_dict['layers'])

    dot_args = dotdict(args_dict)

    dot_args.layers = {}

    for layer_idx in range(n_layers):
        dot_args.layers[layer_idx] = dotdict(args_dict['layers'][str(layer_idx)])

    # Maybe the arg file is missing some newer ones. In that case, give
    # them the default value
    # TODO

    return dot_args


def print_and_save_args(args, path):
    print(args)
    save_args(args, path)


def load_model_from_file(path):
    old_args = load_args(path)

    from common.modular import QStack

    # create model
    model = QStack(old_args)

    load_model(model, os.path.join(path, 'gen.pth'))

    return model, old_args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_model(model, path):
    # load weights
    params = torch.load(path)

    for name, param in params.items():
        if 'buffer' in name.lower():
            if 'reg' in name.lower():
                model.reg_buffer.expand(param.size(0))
            else:
                parts = name.split('.')
                block_id  = int(parts[1])
                model.blocks[block_id].buffer.expand(param.size(0))

    xx = 1

    model.load_state_dict(params)


# loss functions
# ---------------------------------------------------------------------------------

def logistic_ll(mean, logscale, sample, binsize=1 / 256.0):
    # actually discretized logistic, but who cares
    scale = torch.exp(logscale)
    sample = (torch.floor(sample / binsize) * binsize - mean) / scale
    logp = torch.log(torch.sigmoid(sample + binsize / scale) - torch.sigmoid(sample) + 1e-7)
    return logp.sum(dim=(1,2,3))

def gaussian_ll(mean, logscale, sample):
    logscale = logscale.expand_as(mean)
    dist = D.Normal(mean, torch.exp(logscale))
    logp = dist.log_prob(sample)
    return logp.sum(dim=(1,2,3))


if __name__ == '__main__':
    model = load_model_from_file('runs_rebuttal/EGU0_NB2_Comp13.71^27.43^_Coef2.60_4429')
    import pdb; pdb.set_trace()
