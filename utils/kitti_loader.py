import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import sys
import os
import _pickle as pickle

from PIL import Image
from collections import OrderedDict

try:
    from utils.kitti_utils import *
except:
    pass


class preprocessed_kitti(data.Dataset):
    """ load the preprocessed version of the dataset """
    def __init__(self, path, xyz=False):
        self.path = path
        self.data = np.load(path)
        self.xyz  = xyz

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, i):
        try:
            data = self.data #np.load(self.path)
            out = data[str(i)]
        except Exception as e:
            #raise e
            # reload
            self.data = np.load(self.path)
            return self.__getitem__(i)

        if self.xyz:
            out = from_polar_np(out[None])[0]

        return out, np.array([0]).squeeze(0), i

