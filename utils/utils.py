import os
import sys
import pdb
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

# data
# ---------------------------------------------------------------------------------
def to_polar(velo):
    if len(velo.shape) == 4:
        velo = velo.permute(1, 2, 3, 0)

    if velo.shape[2] > 4:
        assert velo.shape[0] <= 4
        velo = velo.permute(1, 2, 0, 3)
        switch=True
    else:
        switch=False

    # assumes r x n/r x (3,4) velo
    dist = torch.sqrt(velo[:, :, 0] ** 2 + velo[:, :, 1] ** 2)
    out = torch.stack([dist, velo[:, :, 2]], dim=2)

    if switch:
        out = out.permute(2, 0, 1, 3)

    if len(velo.shape) == 4:
        out = out.permute(3, 0, 1, 2)

    return out


def from_polar(velo):

    assert velo.ndim == 4, 'expects BS x C x H x W tensor'
    assert int(velo.size(1)) in [2,3], 'second axis must be for channels'

    if velo.size(1) == 3:
        # already in xyz
        return velo

    angles = np.linspace(0, np.pi * 2, velo.shape[-1])
    dist, z = velo[:, 0], velo[:, 1]

    x = torch.Tensor(np.cos(angles)).cuda().unsqueeze(0).unsqueeze(0) * dist
    y = torch.Tensor(np.sin(angles)).cuda().unsqueeze(0).unsqueeze(0) * dist
    out = torch.stack([x,y,z], dim=1)

    return out


def get_chamfer():
    from chamfer_distance.chamfer_distance import ChamferDistance

    chamfer_raw = ChamferDistance()
    prepro      = lambda x : x.reshape(x.size(0), 3, -1).transpose(-2, -1)

    chamfer = lambda x, y : chamfer_raw(prepro(from_polar(x)), prepro(from_polar(y)))[:2]

    return chamfer



# logging
# ---------------------------------------------------------------------------------

class RALog():
    """ keeps track of running averages of values """

    def __init__(self):
        self.reset()

    def reset(self):
        self.storage  = OD()
        self.count    = OD()

    def avg_dict(self, prefix=''):
        out = {}
        for key in self.storage.keys():
            avg = self.storage[key]
            out[prefix + key] = avg

        return out

    def log(self, key, value):
        if 'tensor' in str(type(value)).lower():
            if value.numel() == 1:
                value = value.item()
            else:
                value = value.cpu().data.numpy()

        if key not in self.storage.keys():
            self.count[key] = 1
            self.storage[key] = value
        else:
            prev = self.storage[key]
            cnt  = self.count[key]
            self.storage[key] = (prev * cnt  + value) / (cnt + 1.)
            self.count[key] += 1


# dict wrappers
# ---------------------------------------------------------------------------------

def dict_cat(all_dicts, copy=False, discard=[]):
    assert len(all_dicts) > 1
    ref = all_dicts[0]
    if copy: ref = deepcopy(ref)

    for adict in all_dicts[1:]:
        B_ = None
        for key, value in adict.items():
            if key in discard:
                continue
            if B_ is None:
                B_ = ref[key].size(0)
            if isinstance(ref[key], int):
                ref[key] = value.new(B_).fill_(ref[key])

            ref[key] = torch.cat((ref[key], value))

    return ref


def dict_split(block_dicts, suffix, lens):
    for adict in [x for x in block_dicts.values()]:
        for key in [x for x in adict.keys()]:
            value = adict[key]

            if value.ndim == 0:
                continue

            for suf, len_ in zip(suffix, lens):
                new_key = key + suf
                new_val = value[:len_]
                value   = value[len_:]
                adict[new_key] = new_val

            assert value.size(0) == 0

    return block_dicts


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Model
# ---------------------------------------------------------------------------------

def load_model(model, path):
    # load weights
    params = torch.load(path)

    named_params = {x:y for (x,y) in model.named_parameters()}
    named_params.update({x:y for (x,y) in model.named_buffers()})

    for name, param in params.items():
        if 'buffer' in name.lower():
            if 'dummy' in name.lower():
                model.dummy.buffer.expand(param.size(0))
            else:
                parts = name.split('.')
                block_id  = int(parts[1])
                model.blocks[block_id].buffer.expand(param.size(0))

        if 'quantize' in name:
            # potentially reduce
            n_embeds = param.shape[1]
            if n_embeds != named_params[name].shape[1]:
                block_id = int(name.split('.')[1])
                model.blocks[block_id].quantize.trim(n_embeds=n_embeds)

    if sum('ema_decoder' in x for x in named_params) > 0:
        for block in model.blocks:
            block.ema_decoder = deepcopy(block.decoder)

    model.load_state_dict(params)
    print('successfully loaded model')


# Misc
# ---------------------------------------------------------------------------------

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
