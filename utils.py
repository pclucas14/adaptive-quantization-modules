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
from addict import Dict

# good'ol utils
# ---------------------------------------------------------------------------------
class RALog():
    """ keeps track of running averages of values """

    def __init__(self):
        self.reset()

    def reset(self):
        self.storage  = OD()
        self.count    = OD()
        self.per_task = OD()

    def one_liner(self):
        fill = lambda x, y : (x + ' ' * max(0, y - len(x)))[-y:]
        out = ''
        for key, value in self.storage.items():
            out += fill(key, 20)
            if type(value) == np.ndarray:
                out += str(value) #[:np.argwhere(value == 0)[0][0]])
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


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def print_and_save_args(args, path):
    print(args)
    # let's save the args as json to enable easy loading
    with open(os.path.join(path, 'args.json'), 'w') as f:
        json.dump(vars(args), f)


def load_model_from_file(path):
    with open(os.path.join(path, 'args.json'), 'rb') as f:
        args = dotdict(json.load(f))

    from vqvae import VQVAE
    # create model
    model = VQVAE(args)

    # load weights
    model.load_state_dict(torch.load(os.path.join(path, 'best_model.pth')))

    return model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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

        '''
        avg = 0.
        for key in current.keys():
            avg += current[key]
        avg /= len(current.keys())

        avgs[super_key] = avg
        '''

    return avgs


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
    dd = Log()
    import pdb; pdb.set_trace()
