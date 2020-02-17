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


class Kitti_Img(data.Dataset):

    def get_src_data(split, root='/home/ml/lpagec/pytorch/prednet/kitti_data/numpy_versions'):
        return np.load(os.path.join(root, 'sources_' + split + '.npy')), \
               np.load(os.path.join(root, 'X_' + split + '.npy'))

    train_s, train_data = get_src_data('train')
    val_s,   val_data   = get_src_data('val')
    test_s,  test_data  = get_src_data('test')

    all_src  = np.concatenate((train_s, val_s, test_s))
    all_data = torch.Tensor(np.concatenate((train_data, val_data, test_data))).float()
    print(all_data.shape)

    # all_src  = np.concatenate((val_s, test_s))
    # all_data = torch.Tensor(np.concatenate((val_data, test_data))).float()
    all_data = all_data.permute(0, 3, 1, 2).contiguous()
    unique_src  = np.unique(all_src)

    def __init__(self, args, task_id=-1):
        """ assumes the same structure as when downloaded from the official site """
        """ root should point to the directory containing city/residential/road  """

        self.args = args

        task_name = Kitti_Img.unique_src[task_id]
        idx       = np.argwhere(Kitti_Img.all_src == task_name).squeeze()
        data      = Kitti_Img.all_data[idx]

        self.data = data
        self.rescale = lambda x : (x / 255. - 0.5) * 2.

    def __getitem__(self, index):
        return self.rescale(self.data[index]), np.array([0]).squeeze(0), index

    def __len__(self):
        return len(self.data)
