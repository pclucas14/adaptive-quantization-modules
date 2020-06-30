import os
import pdb
import sys
import torch
import numpy as np
import pickle as pkl
from PIL import Image
from copy import deepcopy
from random import shuffle
from torchvision import datasets, transforms

""" Template Dataset with Labels """
class XYDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, **kwargs):
        self.x, self.y = x, y

        self.rescale = lambda x : (x / 255. - 0.5) * 2.

        # this was to store the inverse permutation in permuted_mnist
        # so that we could 'unscramble' samples and plot them
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        if type(x) != torch.Tensor:
            # mini_imagenet
            # we assume it's a path --> load from file
            x = self.transform(Image.open(x).convert('RGB'))
            y = torch.Tensor(1).fill_(y).long().squeeze()
        else:
            x = x.float() / 255.
            y = y.long()


        # for some reason mnist does better \in [0,1] than [-1, 1]
        if self.source == 'mnist':
            return x, y
        else:
            return (x - .5) * 2, y, idx


class Kitti_dataset(torch.utils.data.Dataset):
    def __init__(self, paths, hist_equalize=True):
        self.paths = paths
        self.datas = [np.load(path) for path in paths]
        self.lens  = [len(x) for x in self.datas]
        self.cumlens = np.cumsum(self.lens)

        self.len = sum(self.lens)

        self.eq = hist_equalize
        self.hist = KITTI_HIST
        self.bins = BINS
        self.cdf = self.hist.cumsum()
        self.cdf = 255 * self.cdf / self.cdf[-1]

    def unnorm(self, x):

        dev = None
        if type(x) == torch.Tensor:
            dev = x.device
            x = x.cpu().data.numpy()

        x = np.interp(x.flatten(), self.cdf, self.bins[:-1]).reshape(x.shape)

        if dev is not None:
            x = torch.from_numpy(x).to(dev).float()

        return x * .5 + .5

    def norm(self, x):
        x = np.interp(x.flatten(), self.bins[:-1], self.cdf).reshape(x.shape).astype(np.float32)
        return (x - .5) * 2.


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        rec_idx = np.argwhere((idx < self.cumlens) == 1).min()
        sample_idx = idx - (self.cumlens[rec_idx - 1] if rec_idx > 0 else 0)

        item = self.datas[rec_idx]['%d.npy' % sample_idx]

        if self.eq:
            item = self.norm(item)

        # TODO: should we map back and forth from polar to xyz ?
        return item,  0, idx


""" Template Dataset for Continual Learning """
class CLDataLoader(object):
    def __init__(self, datasets_per_task, args, train=True):
        test_bs, num_workers = 128, 8

        if 'kitti' in args.dataset:
            test_bs = 32
            num_workers = 0
        elif 'imagenet' in args.dataset:
            test_bs = 32

        bs = args.batch_size if train else test_bs
        if args.debug: num_workers = 0

        self.datasets = datasets_per_task
        self.loaders = [
                torch.utils.data.DataLoader(x, batch_size=bs, shuffle=True, drop_last=train,
                    num_workers=num_workers) for x in self.datasets ]

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)


def get_processed_kitti(args, mode='offline'):
    # from utils.kitti_loader import preprocessed_kitti

    root = '../datasets/processed_kitti'

    task_id = 0
    env_recs = {}
    for env in os.listdir(root):
        env_recs[env] = []
        for recording in os.listdir(os.path.join(root, env)):
            path = os.path.join(root, env, recording, 'processed.npz')
            env_recs[env] += [path]
            task_id += 1

    if mode == 'offline':
        # train on residential and road
        train_recs = env_recs['road'][2:] + env_recs['residential'][3:]
        valid_recs = env_recs['road'][:2] + env_recs['residential'][:3]

    elif mode == 'online':
        train_recs = env_recs['city'][2:]
        valid_recs = env_recs['city'][:2]

    elif mode == 'all':
        all_recs = env_recs['road'] + env_recs['residential'] + env_recs['city']
        train_recs = all_recs
        valid_recs = all_recs

    train_ds = Kitti_dataset(train_recs)
    valid_ds = Kitti_dataset(valid_recs)

    return [train_ds], [valid_ds], [valid_ds]



def get_split_cifar10(args):
    # assert args.n_tasks in [5, 10], 'SplitCifar only works with 5 or 10 tasks'
    assert '1.' in str(torch.__version__)[:2], 'Use Pytorch 1.x!'
    args.n_tasks   = 5
    args.n_classes = 10
    args.multiple_heads = False
    args.input_size = (3, 32, 32)

    if args.override_cl_defaults:
        pass
    else:
        args.n_classes_per_task = 2

    args.n_tasks = args.n_classes // args.n_classes_per_task

    # fetch data
    train = datasets.CIFAR10('../cl-pytorch/data/', train=True,  download=True)
    test  = datasets.CIFAR10('../cl-pytorch/data/', train=False, download=True)

    train_x, train_y = train.data, train.targets
    test_x,  test_y  = test.data,  test.targets

    # sort according to the label
    out_train = [
        (x,y) for (x,y) in sorted(zip(train_x, train_y), key=lambda v : v[1]) ]

    out_test = [
        (x,y) for (x,y) in sorted(zip(test_x, test_y), key=lambda v : v[1]) ]

    train_x, train_y = [
            np.stack([elem[i] for elem in out_train]) for i in [0,1] ]

    test_x,  test_y  = [
            np.stack([elem[i] for elem in out_test]) for i in [0,1] ]

    train_x = torch.Tensor(train_x).permute(0, 3, 1, 2).contiguous()
    test_x  = torch.Tensor(test_x).permute(0, 3, 1, 2).contiguous()

    train_y = torch.Tensor(train_y)
    test_y  = torch.Tensor(test_y)

    # get indices of class split
    train_idx = [((train_y + i) % 10).argmax() for i in range(10)]
    train_idx = [0] + [x + 1 for x in sorted(train_idx)]

    test_idx  = [((test_y + i) % 10).argmax() for i in range(10)]
    test_idx  = [0] + [x + 1 for x in sorted(test_idx)]

    train_ds, valid_ds, test_ds = [], [], []
    skip = args.n_classes_per_task
    for i in range(0, 10, skip):
        tr_s, tr_e = train_idx[i], train_idx[i + skip]
        te_s, te_e = test_idx[i],  test_idx[i + skip]

        train_ds += [(train_x[tr_s:tr_e], train_y[tr_s:tr_e])]
        test_ds  += [(test_x[te_s:te_e],  test_y[te_s:te_e])]

    train_ds, valid_ds = make_valid_from_train(train_ds)

    train_ds  = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), train_ds)
    valid_ds  = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), valid_ds)
    test_ds   = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), test_ds)

    return train_ds, valid_ds, test_ds


def get_split_cifar100(args):
    # assert args.n_tasks in [5, 10], 'SplitCifar only works with 5 or 10 tasks'
    assert '1.' in str(torch.__version__)[:2], 'Use Pytorch 1.x!'
    args.n_tasks   = 20
    args.n_classes = 100
    args.input_size = (3, 32, 32)

    if args.override_cl_defaults:
        print('overriding default values')
        print('multiple heads :      {}'.format(args.multiple_heads))
        print('n classes per task :  {}'.format(args.n_classes_per_task))
        assert args.multiple_heads > 0 and args.n_classes_per_task > 0
    else:
        args.multiple_heads = True
        args.n_classes_per_task = 5

    # fetch data
    train = datasets.CIFAR100('../../cl-pytorch/data/', train=True,  download=True)
    test  = datasets.CIFAR100('../../cl-pytorch/data/', train=False, download=True)

    train_x, train_y = train.data, train.targets
    test_x,  test_y  = test.data,  test.targets

    # sort according to the label
    out_train = [
        (x,y) for (x,y) in sorted(zip(train_x, train_y), key=lambda v : v[1]) ]

    out_test = [
        (x,y) for (x,y) in sorted(zip(test_x, test_y), key=lambda v : v[1]) ]

    train_x, train_y = [
            np.stack([elem[i] for elem in out_train]) for i in [0,1] ]

    test_x,  test_y  = [
            np.stack([elem[i] for elem in out_test]) for i in [0,1] ]

    train_x = torch.Tensor(train_x).permute(0, 3, 1, 2).contiguous()
    test_x  = torch.Tensor(test_x).permute(0, 3, 1, 2).contiguous()

    train_y = torch.Tensor(train_y)
    test_y  = torch.Tensor(test_y)

    # get indices of class split
    train_idx = [((train_y + i) % 100).argmax() for i in range(100)]
    train_idx = [0] + [x + 1 for x in sorted(train_idx)]

    test_idx  = [((test_y + i) % 100).argmax() for i in range(100)]
    test_idx  = [0] + [x + 1 for x in sorted(test_idx)]

    train_ds, valid_ds, test_ds = [], [], []
    skip = 1 # get all classes individually first
    for i in range(0, 100):
        tr_s, tr_e = train_idx[i], train_idx[i + skip]
        te_s, te_e = test_idx[i],  test_idx[i + skip]

        train_size = tr_e - tr_s
        split = tr_s + int(0.9 * train_size.item())

        train_ds += [(train_x[tr_s:split], train_y[tr_s:split])]
        valid_ds += [(train_x[split:tr_e], train_y[split:tr_e])]
        test_ds  += [(test_x[te_s:te_e],  test_y[te_s:te_e])]

    # next we randomly partition the dataset
    indices = [x for x in range(100)]

    # note: previously we shuffled the indices to make the split
    # random. However we left it out to be consistent with A-GEM

    train_classes = [train_ds[indices[i]] for i in range(100)]
    valid_classes = [valid_ds[indices[i]] for i in range(100)]
    test_classes  = [test_ds[indices[i]]  for i in range(100)]

    train_ds, valid_ds, test_ds = [], [], []

    skip = args.n_classes_per_task
    for i in range(0, 100, skip):
        train_task_ds, valid_task_ds, test_task_ds = [[], []], [[], []], [[], []]
        for j in range(skip):
            train_task_ds[0] += [train_classes[i + j][0]]
            train_task_ds[1] += [train_classes[i + j][1]]
            valid_task_ds[0] += [valid_classes[i + j][0]]
            valid_task_ds[1] += [valid_classes[i + j][1]]
            test_task_ds[0]  += [test_classes[i + j][0]]
            test_task_ds[1]  += [test_classes[i + j][1]]

        train_ds += [(torch.cat(train_task_ds[0]), torch.cat(train_task_ds[1]))]
        valid_ds += [(torch.cat(valid_task_ds[0]), torch.cat(valid_task_ds[1]))]
        test_ds  += [(torch.cat(test_task_ds[0]), torch.cat(test_task_ds[1]))]

    # TODO: remove this
    # Facebook actually does 17 tasks (3 to CV)
    train_ds = train_ds[:args.n_tasks]
    valid_ds = valid_ds[:args.n_tasks]
    test_ds  = test_ds[:args.n_tasks]

    # build masks
    masks = []
    task_ids = [None for _ in range(args.n_tasks)]
    for task, task_data in enumerate(train_ds):
        labels = task_data[1].unique().long()
        assert labels.shape[0] == args.n_classes_per_task
        mask = torch.zeros(100).to(args.device)
        mask[labels] = 1
        masks += [mask]
        task_ids[task] = labels

    task_ids = torch.stack(task_ids).to(args.device).long()

    train_ds = map(lambda x, y : XYDataset(x[0], x[1], **{'source':'cifar100', 'mask':y, 'task_ids':task_ids}), train_ds, masks)
    valid_ds = map(lambda x, y : XYDataset(x[0], x[1], **{'source':'cifar100', 'mask':y, 'task_ids':task_ids}), valid_ds, masks)
    test_ds  = map(lambda x, y : XYDataset(x[0], x[1], **{'source':'cifar100', 'mask':y, 'task_ids':task_ids}), test_ds, masks)

    return train_ds, valid_ds, test_ds


def get_miniimagenet(args):
    ROOT_PATH = '/home/eugene/data/filelists/miniImagenet/materials/images'
    ROOT_PATH_CSV = '/home/eugene/data/filelists/miniImagenet/materials'
    ROOT_PATH = '../cl-pytorch/data/imagenet/imagenet_images'
    ROOT_PATH_CSV = '../prototypical-network-pytorch/materials'
    ROOT_PATH = '/private/home/lucaspc/repos/datasets/miniimagenet/images'
    ROOT_PATH_CSV = '/private/home/lucaspc/repos/datasets/miniimagenet/splits'



    size = args.data_size[-1]
    args.n_classes = 100
    args.input_size = args.data_size

    if args.override_cl_defaults:
        print('overriding default values')
        print('multiple heads :      {}'.format(args.multiple_heads))
        print('n classes per task :  {}'.format(args.n_classes_per_task))
        assert args.multiple_heads > -1 and args.n_classes_per_task > -1
    else:
        args.multiple_heads = True
        args.n_classes_per_task = 5

    args.n_tasks = args.n_classes // args.n_classes_per_task


    def get_data(setname):
        csv_path = os.path.join(ROOT_PATH_CSV, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = os.path.join(ROOT_PATH, name)
            if wnid not in wnids:
                wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        return data, label


    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    train_data, train_label = get_data('train')
    valid_data, valid_label = get_data('val')
    test_data,  test_label  = get_data('test')

    train_amt = np.unique(train_label).shape[0]
    valid_amt = np.unique(valid_label).shape[0]

    valid_label = [x + train_amt for x in valid_label]
    test_label =  [x + train_amt + valid_amt for x in test_label]

    # total of 60k examples for training, the rest for testing
    all_data  = np.array(train_data  + valid_data  + test_data)
    all_label = np.array(train_label + valid_label + test_label)

    # TODO: remove this
    # all_data = all_data[::-1]
    # all_label = all_label[::-1]


    train_ds, valid_ds, test_ds = [], [], []
    current_train, current_val, current_test = None, None, None

    cat = lambda x, y: np.concatenate((x, y), axis=0)

    for i in range(args.n_classes):
        class_indices = np.argwhere(all_label == i).reshape(-1)
        class_data  = all_data[class_indices]
        class_label = all_label[class_indices]
        split   = int(0.8 * class_data.shape[0])
        split_b = int(0.9 * class_data.shape[0])

        data_train, data_valid, data_test = class_data[:split], class_data[split:split_b], class_data[split_b:]
        label_train, label_valid, label_test = class_label[:split], class_label[split:split_b], class_label[split_b:]

        if current_train is None:
            current_train, current_valid, current_test = (data_train, label_train), (data_valid, label_valid), (data_test, label_test)
        else:
            current_train = cat(current_train[0], data_train), cat(current_train[1], label_train)
            current_valid = cat(current_valid[0], data_valid), cat(current_valid[1], label_valid)
            current_test  = cat(current_test[0],  data_test),  cat(current_test[1],  label_test)

        if i % args.n_classes_per_task == (args.n_classes_per_task  - 1):
            train_ds += [current_train]
            valid_ds += [current_valid]
            test_ds  += [current_test]
            current_train, current_valid, current_test = None, None, None

    # TODO: remove this
    # Facebook actually does 17 tasks (3 to CV)
    train_ds = train_ds[:args.n_tasks]
    valid_ds = valid_ds[:args.n_tasks]
    test_ds  = test_ds[:args.n_tasks]

    # build masks
    masks = []
    task_ids = [None for _ in range(args.n_tasks)]
    for task, task_data in enumerate(train_ds):
        labels = np.unique(task_data[1])
        assert labels.shape[0] == args.n_classes_per_task
        mask = torch.zeros(args.n_classes).to(args.device)
        mask[labels] = 1
        masks += [mask]
        task_ids[task] = labels

    task_ids = torch.from_numpy(np.stack(task_ids)).to(args.device).long()

    train_ds  = map(lambda x, y : XYDataset(x[0], x[1], **{'source':'cifar100', 'mask':y, 'task_ids':task_ids, 'transform':transform}), train_ds, masks)
    valid_ds  = map(lambda x, y : XYDataset(x[0], x[1], **{'source':'cifar100', 'mask':y, 'task_ids':task_ids, 'transform':transform}), valid_ds, masks)
    test_ds   = map(lambda x, y : XYDataset(x[0], x[1], **{'source':'cifar100', 'mask':y, 'task_ids':task_ids, 'transform':transform}), test_ds, masks)

    return train_ds, valid_ds, test_ds


def make_valid_from_train(dataset, cut=0.9):
    tr_ds, val_ds = [], []
    for task_ds in dataset:
        x_t, y_t = task_ds

        # first we shuffle
        perm = torch.randperm(x_t.size(0))
        x_t, y_t = x_t[perm], y_t[perm]

        split = int(x_t.size(0) * cut)
        x_tr, y_tr   = x_t[:split], y_t[:split]
        x_val, y_val = x_t[split:], y_t[split:]

        tr_ds  += [(x_tr, y_tr)]
        val_ds += [(x_val, y_val)]

    return tr_ds, val_ds


KITTI_HIST = np.\
array([5.27350761e-02, 5.84365022e-02, 1.56835853e-01, 5.12492286e-01,
       7.49505187e-01, 4.02209283e-01, 2.36387599e-01, 1.84520214e-01,
       1.75269423e-01, 1.77146950e-01, 1.54836535e-01, 1.24862912e-01,
       9.87950089e-02, 6.45945846e-02, 2.94048049e-02, 5.13773398e-03,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       3.44969743e-02, 2.18942202e-02, 2.03443101e-02, 2.09122401e-02,
       2.21134924e-02, 2.18244409e-02, 2.22442881e-02, 2.29772412e-02,
       2.40840115e-02, 2.56813113e-02, 2.94629861e-02, 3.62847301e-02,
       5.11401021e-02, 6.85468571e-02, 7.70378654e-02, 8.32589421e-02,
       8.79590393e-02, 8.64607407e-02, 8.76696710e-02, 9.63377523e-02,
       9.14953729e-02, 8.05629425e-02, 7.45737923e-02, 6.94589249e-02,
       6.51718315e-02, 6.24376763e-02, 5.76867525e-02, 5.34318049e-02,
       5.02411311e-02, 4.78692351e-02, 4.69167352e-02, 4.66565581e-02,
       4.46727175e-02, 4.26360218e-02, 4.12217138e-02, 4.08088330e-02,
       4.04924565e-02, 3.92887216e-02, 3.70670802e-02, 3.45278214e-02,
       3.29156846e-02, 3.19180386e-02, 3.15119889e-02, 3.03920911e-02,
       2.92124441e-02, 2.78548438e-02, 2.65230880e-02, 2.54417185e-02,
       2.44591028e-02, 2.37319532e-02, 2.31331312e-02, 2.23046966e-02,
       2.18417612e-02, 2.11947171e-02, 2.04596316e-02, 1.97346208e-02,
       1.90764784e-02, 1.86725561e-02, 1.82604763e-02, 1.79877965e-02,
       1.78123948e-02, 1.74356270e-02, 1.70739537e-02, 1.67243583e-02,
       1.62644222e-02, 1.58431680e-02, 1.54139823e-02, 1.48829391e-02,
       1.44081108e-02, 1.43644264e-02, 1.41479255e-02, 1.38763138e-02,
       1.34984194e-02, 1.35671729e-02, 1.31285927e-02, 1.28693452e-02,
       1.25968642e-02, 1.24908676e-02, 1.23250397e-02, 1.15436222e-02,
       1.12729356e-02, 1.10061252e-02, 1.07599366e-02, 1.03412489e-02,
       1.01371483e-02, 9.69273501e-03, 9.35260469e-03, 9.05326214e-03,
       8.86141434e-03, 8.82543719e-03, 8.67734294e-03, 8.16427308e-03,
       7.86702883e-03, 7.77116561e-03, 7.56074926e-03, 7.42610663e-03,
       7.27605822e-03, 6.82185227e-03, 6.72339261e-03, 6.77003159e-03,
       6.31941067e-03, 6.07810187e-03, 5.95467382e-03, 5.77562072e-03,
       5.76807312e-03, 5.88554726e-03, 5.25016105e-03, 5.14738891e-03,
       5.08610389e-03, 5.11983900e-03, 4.95517004e-03, 4.89976452e-03,
       4.91381347e-03, 4.39446515e-03, 4.36033088e-03, 4.28328814e-03,
       4.29058877e-03, 4.42082502e-03, 3.90626775e-03, 3.74429862e-03,
       3.67568686e-03, 3.61540163e-03, 3.61711394e-03, 3.67405663e-03,
       3.70563176e-03, 3.07414395e-03, 2.99029774e-03, 2.98902837e-03,
       2.98123531e-03, 2.94934681e-03, 2.95218573e-03, 2.94595949e-03,
       2.94899614e-03, 2.45633224e-03, 2.44210049e-03, 2.39797195e-03,
       2.35388465e-03, 2.30831259e-03, 2.33814185e-03, 2.35276922e-03,
       2.37604395e-03, 2.59340653e-03, 1.95066616e-03, 1.93754349e-03,
       1.88119769e-03, 1.88317114e-03, 1.85758722e-03, 1.94594090e-03,
       1.87305023e-03, 1.82138250e-03, 1.80069355e-03, 1.92996680e-03,
       1.34420773e-03, 1.33967515e-03, 1.35061677e-03, 1.26404628e-03,
       1.23565701e-03, 1.22056715e-03, 1.19959032e-03, 1.17296932e-03,
       1.18063554e-03, 1.21825039e-03, 1.27883034e-03, 1.19419972e-03,
       8.55886156e-04, 8.57588052e-04, 8.67253056e-04, 8.62403378e-04,
       8.50156075e-04, 8.10836375e-04, 8.70050948e-04, 8.32335372e-04,
       8.50350062e-04, 8.63247185e-04, 9.00671071e-04, 1.01781692e-03,
       5.77652649e-04, 5.82118083e-04, 5.71877801e-04, 5.24007745e-04,
       5.12476702e-04, 5.16544160e-04, 5.20829341e-04, 4.99043093e-04,
       4.97879170e-04, 5.07600910e-04, 5.34460667e-04, 5.46018655e-04,
       5.93622760e-04, 5.94335786e-04, 3.03153344e-04, 2.93166957e-04,
       2.95875283e-04, 2.95218428e-04, 2.90365369e-04, 2.75860515e-04,
       2.70004428e-04, 2.67787568e-04, 2.72895548e-04, 2.62281251e-04,
       2.79666363e-04, 2.69985776e-04, 2.68541142e-04, 2.67482626e-04,
       2.79594813e-04, 2.92648421e-04, 3.27185657e-04, 4.21204372e-04,
       1.37288996e-04, 1.37848185e-04, 1.40922480e-04, 1.45559267e-04,
       1.43683033e-04, 1.41783933e-04, 1.46454777e-04, 1.34102696e-04,
       1.37874681e-04, 1.40959785e-04, 1.39456059e-04, 1.51222326e-04,
       1.64122569e-04, 1.79719436e-04, 1.95810006e-04, 3.06384035e-04])

BINS = np.\
array([-2.399     , -2.2419238 , -2.0848477 , -1.9277716 , -1.7706954 ,
       -1.6136193 , -1.4565432 , -1.2994671 , -1.142391  , -0.9853148 ,
       -0.82823867, -0.67116255, -0.5140864 , -0.35701028, -0.19993415,
       -0.04285803,  0.1142181 ,  0.27129424,  0.42837036,  0.5854465 ,
        0.7425226 ,  0.8995987 ,  1.0566748 ,  1.213751  ,  1.3708271 ,
        1.5279032 ,  1.6849793 ,  1.8420554 ,  1.9991317 ,  2.1562078 ,
        2.313284  ,  2.47036   ,  2.6274362 ,  2.7845123 ,  2.9415884 ,
        3.0986645 ,  3.2557406 ,  3.4128168 ,  3.569893  ,  3.726969  ,
        3.8840451 ,  4.0411215 ,  4.1981974 ,  4.3552737 ,  4.5123496 ,
        4.669426  ,  4.826502  ,  4.983578  ,  5.140654  ,  5.2977304 ,
        5.4548063 ,  5.6118827 ,  5.7689586 ,  5.926035  ,  6.083111  ,
        6.240187  ,  6.397263  ,  6.5543394 ,  6.7114153 ,  6.8684916 ,
        7.0255675 ,  7.182644  ,  7.33972   ,  7.496796  ,  7.653872  ,
        7.8109484 ,  7.9680243 ,  8.1251    ,  8.282177  ,  8.439253  ,
        8.596329  ,  8.753405  ,  8.910481  ,  9.067557  ,  9.224633  ,
        9.38171   ,  9.538786  ,  9.695862  ,  9.852938  , 10.010015  ,
       10.16709   , 10.324166  , 10.481242  , 10.638319  , 10.795395  ,
       10.952471  , 11.109547  , 11.2666235 , 11.423699  , 11.580775  ,
       11.737851  , 11.894928  , 12.052004  , 12.20908   , 12.366156  ,
       12.523232  , 12.680308  , 12.837384  , 12.99446   , 13.151537  ,
       13.308613  , 13.465689  , 13.622765  , 13.779841  , 13.936917  ,
       14.093993  , 14.251069  , 14.408146  , 14.565222  , 14.722298  ,
       14.8793745 , 15.03645   , 15.193526  , 15.350602  , 15.507679  ,
       15.664755  , 15.821831  , 15.978907  , 16.135983  , 16.293058  ,
       16.450136  , 16.607212  , 16.764288  , 16.921364  , 17.07844   ,
       17.235516  , 17.392591  , 17.549667  , 17.706745  , 17.863821  ,
       18.020897  , 18.177973  , 18.335049  , 18.492125  , 18.6492    ,
       18.806276  , 18.963354  , 19.12043   , 19.277506  , 19.434582  ,
       19.591658  , 19.748734  , 19.90581   , 20.062885  , 20.219963  ,
       20.377039  , 20.534115  , 20.69119   , 20.848267  , 21.005342  ,
       21.162418  , 21.319496  , 21.476572  , 21.633648  , 21.790724  ,
       21.9478    , 22.104876  , 22.261951  , 22.419027  , 22.576105  ,
       22.733181  , 22.890257  , 23.047333  , 23.204409  , 23.361485  ,
       23.51856   , 23.675636  , 23.832714  , 23.98979   , 24.146866  ,
       24.303942  , 24.461018  , 24.618093  , 24.77517   , 24.932245  ,
       25.089323  , 25.246399  , 25.403475  , 25.56055   , 25.717627  ,
       25.874702  , 26.031778  , 26.188854  , 26.345932  , 26.503008  ,
       26.660084  , 26.81716   , 26.974236  , 27.131311  , 27.288387  ,
       27.445465  , 27.602541  , 27.759617  , 27.916693  , 28.073769  ,
       28.230844  , 28.38792   , 28.544996  , 28.702074  , 28.85915   ,
       29.016226  , 29.173302  , 29.330378  , 29.487453  , 29.64453   ,
       29.801605  , 29.958683  , 30.115759  , 30.272835  , 30.42991   ,
       30.586987  , 30.744062  , 30.901138  , 31.058214  , 31.215292  ,
       31.372368  , 31.529444  , 31.68652   , 31.843596  , 32.00067   ,
       32.15775   , 32.314823  , 32.4719    , 32.628975  , 32.786053  ,
       32.94313   , 33.100204  , 33.257282  , 33.414356  , 33.571434  ,
       33.728508  , 33.885586  , 34.04266   , 34.199738  , 34.356815  ,
       34.51389   , 34.670967  , 34.82804   , 34.98512   , 35.142193  ,
       35.29927   , 35.45635   , 35.613422  , 35.7705    , 35.927574  ,
       36.084652  , 36.241726  , 36.398804  , 36.555878  , 36.712955  ,
       36.870033  , 37.027107  , 37.184185  , 37.34126   , 37.498337  ,
       37.65541   , 37.81249   ])

if __name__ == '__main__':
    class args:
        pass

    ds = get_processed_kitti(args, mode='all')[0][0]
    dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=0)

    pdb.set_trace()

    full = []
    for item in dl:
        full += [item[0]]
        print(item[0].shape)

    pdb.set_trace()
    full = torch.cat(full)
    xx = full.cpu().data.numpy()

    import gzip
    f = gzip.GzipFile("city_recs.npy.gz", "w")
    np.save(file=f, arr=xx)
    f.close()
