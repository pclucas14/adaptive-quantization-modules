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

class Kitti(data.Dataset):
    def __init__(self, args, root='/mnt/data/lpagec/kitti_data/raw', n_points_pre=2048, n_points_post=1024, task_id=-1):
        """ assumes the same structure as when downloaded from the official site """
        """ root should point to the directory containing city/residential/road  """

        '''
        n_points_pre  : amt of points used for the preprocessing (used only to build path)
        n_points_post : amt of points to which the loader with downsample
        '''

        assert n_points_pre % n_points_post == 0
        self.pre, self.post = n_points_pre, n_points_post
        self.args = args

        self.normalize = self.args.normalize

        # we need to build a mapper from index to path from which to load
        mapper = []
        task_dict = OrderedDict()
        seen_so_far = []
        task = 0

        task_dict[task] = []
        for dir_name, subdir_list, file_list in os.walk(root):
            if 'processed%d_velodyne_points/data' % self.pre in dir_name:
                to_be_added = [os.path.join(dir_name, f) for f in file_list]
                mapper += to_be_added

                seen_so_far += [len(mapper)]
                task_dict[task] = to_be_added
                task += 1

        self.mapper = mapper
        self.task_dict = task_dict

        # preprocessing fails for these point clouds. We remove them manually
        remove = [4185, 4297, 4619, 5737, 5738, 7371, 7377, 9651, 9652, 9653, 9654, 9655, 9656, 9657, 9658, 9699, 9700, 9701, 9702, 9703, 9704, 9705, 9706, 9707, 9709, 9710, 9711, 9712, 9713, 15262, 15263, 15281, 17417, 17418, 17448, 17449, 17484, 17485, 18779, 18780, 18782, 18783, 18784, 18785, 18786, 18787, 18788, 18789, 18790, 18791, 18792, 18793, 18794, 18795, 18796, 18797, 18798, 18803, 18905, 18906, 18907, 18908, 18917, 18919, 18920, 18956, 18957, 18958, 18960, 18961, 18962, 18963, 18964, 18965, 18966, 18967, 18968, 18969, 18970, 18971, 18972, 18973, 18974, 18975, 18976, 18977, 18978, 18979, 18980, 19006, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 22619, 22620, 22621, 23438, 23439, 23440, 23441, 23545, 23546, 23547, 23548, 23549, 23550, 23551, 23687, 23688, 23689, 24845, 24846, 24848, 28111, 31748, 31749, 32301, 33587, 35343, 35834, 38519, 38895, 40199, 40200, 40201, 40215, 40216, 40217, 40218, 40219]

        j, ij = 0, 0
        for i, index in enumerate(remove):
            test_a = self.mapper[index - i]
            del self.mapper[index - i]

            # find corresponding task
            while seen_so_far[j] <= index:
                j += 1
                ij = 0

            offset = index - seen_so_far[j]
            test_b = self.task_dict[j][offset] # - ij]
            assert test_a == test_b, 'these two point clouds are supposed to be the same!'

            del self.task_dict[j][offset] # - ij]
            ij += 1

        # reverse the list for now
        task_dict_ = {}
        for key, value in task_dict.items():
            task_dict_[len(task_dict) - key - 1] = value

        xx = [len(task_dict[i]) for i in range(len(task_dict))]
        yy = [len(task_dict_[i]) for i in range(len(task_dict_))]
        self.task_dict = task_dict_

        if task_id != -1:
            self.mapper = self.task_dict[task_id]


    def __getitem__(self, index):
        point_cloud = np.load(self.mapper[index])
        a, b, c = point_cloud.shape
        point_cloud = preprocess(point_cloud.reshape(1, a, b, c), normalize=self.normalize)

        if self.args.xyz:
            point_cloud = from_polar_np(point_cloud)

        out = point_cloud.squeeze()[:, :, ::(self.pre // self.post)]

        return out.astype('float32'), np.array([0]).squeeze(0), index

    def get_og(self, index):
        path = self.mapper[index]
        path = path.replace('processed%d_' % self.pre , '').replace('.npy', '.bin')
        pc   = np.fromfile(path, dtype=np.float32)
        return pc.reshape((-1, 4))[:, :3] / pc.max()

    def get_img(self, index):
        path = self.mapper[index]
        path = path.replace('processed%d_velodyne_points' % self.pre , 'image_03').replace('.npy', '.png')
        return Image.open(path)

    def show_img_stream(self, index, len=10):
        [self.get_img(x).show() for x in range(max(0, index - 10), index)]

    def compare(self, index):
        img = self.get_img(index)
        og = self.get_og(index)
        pc = self.__getitem__(index)
        pc = pc.reshape((1, *pc.shape))
        pc = from_polar_np(pc)
        img.show()
        show_pc(og)
        show_pc(pc)

    def __len__(self):
        return len(self.mapper)


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


if __name__ == '__main__':
    class args:
        pass
    args.xyz = True

    ds = Kitti_Img(args)

    '''
    def show_pc(velo, ind=1, show=True):
        import matplotlib.pyplot as plt
        plt.scatter(velo[:, 0], velo[:, 1], s=0.7, color='k')
        plt.show()

    show = lambda x : show_pc(from_polar_np(x.squeeze().reshape(2, -1).transpose(1,0)))

    xx = ds[1234]
    zz = from_polar_np(xx)
    import pdb; pdb.set_trace()
    ds.compare(len(ds.mapper)-10000)
    # is the preprocess step too much of a bottleneck ?
    dl_loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=10, drop_last=True)


    import time
    s = time.time()
    for i, batch in enumerate(dl_loader):
        print("%d / %d" % (i, len(dl_loader)))
        e = time.time()
        print('{:.4f} seconds'.format(e - s))
        s = time.time()
    '''
