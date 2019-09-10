import os
import sys
import torch
import numpy as np
import pickle as pkl
from PIL import Image
from random import shuffle

from torchvision import datasets, transforms

""" Template Dataset with Labels """
class XYDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, **kwargs):
        self.x, self.y = x, y

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
            return (x - 0.5) * 2, y


""" Template Dataset for Continual Learning """
class CLDataLoader(object):
    def __init__(self, datasets_per_task, args, train=True):
        bs = args.batch_size if train else 32 #256

        self.datasets = datasets_per_task
        self.loaders = [
                torch.utils.data.DataLoader(x, batch_size=bs, shuffle=True, drop_last=train, num_workers=0)
                for x in self.datasets ]

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)


""" Kitti Lidar continual dataset """
def get_kitti(args):
    # get datasets
    args.input_size = (3, 40, 512)
    from kitti_loader import Kitti
    max_task = 61 if args.max_task == -1 else args.max_task
    dss_train = [Kitti(args, task_id=i) for i in range(max_task) if i != 17]
    dss_valid = [Kitti(args, task_id=i) for i in range(max_task) if i != 17]
    dss_test  = [Kitti(args, task_id=i) for i in range(max_task) if i != 17]

    for (ds_tr, ds_val, ds_te) in zip(dss_train, dss_valid, dss_test):
        assert len(ds_tr.mapper) == len(ds_val.mapper) == len(ds_te.mapper)
        len_ = len(ds_tr.mapper)
        split_a, split_b = int(0.8 * len_), int(0.9 * len_)

        ds_tr.mapper  = ds_tr.mapper[:split_a]
        ds_val.mapper = ds_val.mapper[split_a:split_b]
        ds_te.mapper  = ds_te.mapper[split_b:]

    return dss_train, dss_valid, dss_test


""" Permuted MNIST """
def get_permuted_mnist(args):
    assert not args.use_conv

    # fetch MNIST
    train = datasets.MNIST('../cl-pytorch/data/', train=True,  download=True)
    test  = datasets.MNIST('../cl-pytorch/data/', train=False, download=True)

    train_x, train_y = train.train_data, train.train_labels
    test_x,  test_y  = test.test_data,   test.test_labels

    train_x = train_x.view(train_x.size(0), -1)
    test_x  = test_x.view(test_x.size(0), -1)

    train_ds, test_ds, inv_perms = [], [], []
    for task in range(args.n_tasks):
        perm = torch.arange(train_x.size(-1)) if task == 0 else torch.randperm(train_x.size(-1))

        # build inverse permutations, so we can display samples
        inv_perm = torch.zeros_like(perm)
        for i in range(perm.size(0)):
            inv_perm[perm[i]] = i

        inv_perms += [inv_perm]
        train_ds  += [(train_x[:, perm], train_y)]
        test_ds   += [(test_x[:, perm],  test_y)]

    train_ds = map(lambda x, y : XYDataset(x[0], x[1], **{'inv_perm': y, 'source': 'mnist'}), train_ds, inv_perms)
    test_ds  = map(lambda x, y : XYDataset(x[0], x[1], **{'inv_perm': y, 'source': 'mnist'}), test_ds,  inv_perms)

    return train_ds, test_ds


""" Split MNIST into 5 tasks {{0,1}, ... {8,9}} """
def get_split_mnist(args):
    assert args.n_tasks in [5, 10], 'SplitMnist only works with 5 or 10 tasks'
    assert '1.' in str(torch.__version__)[:2], 'Use Pytorch 1.x!'

    # fetch MNIST
    train = datasets.MNIST('../cl-pytorch/data/', train=True,  download=True)
    test  = datasets.MNIST('../cl-pytorch/data/', train=False, download=True)

    train_x, train_y = train.train_data, train.train_labels
    test_x,  test_y  = test.test_data,   test.test_labels

    # sort according to the label
    out_train = [
        (x,y) for (x,y) in sorted(zip(train_x, train_y), key=lambda v : v[1]) ]

    out_test = [
        (x,y) for (x,y) in sorted(zip(test_x, test_y), key=lambda v : v[1]) ]

    train_x, train_y = [
            torch.stack([elem[i] for elem in out_train]) for i in [0,1] ]

    test_x,  test_y  = [
            torch.stack([elem[i] for elem in out_test]) for i in [0,1] ]

    if args.use_conv:
        train_x = train_x.unsqueeze(1)
        test_x  = test_x.unsqueeze(1)
    else:
        train_x = train_x.view(train_x.size(0), -1)
        test_x  = test_x.view(test_x.size(0), -1)

    # get indices of class split
    train_idx = [((train_y + i) % 10).argmax() for i in range(10)]
    train_idx = [0] + sorted(train_idx)

    test_idx  = [((test_y + i) % 10).argmax() for i in range(10)]
    test_idx  = [0] + sorted(test_idx)

    train_ds, test_ds = [], []
    skip = 10 // args.n_tasks
    for i in range(0, 10, skip):
        tr_s, tr_e = train_idx[i], train_idx[i + skip]
        te_s, te_e = test_idx[i],  test_idx[i + skip]

        train_ds += [(train_x[tr_s:tr_e], train_y[tr_s:tr_e])]
        test_ds  += [(test_x[te_s:te_e],  test_y[te_s:te_e])]

    train_ds = map(lambda x : XYDataset(x[0], x[1], **{'source': 'mnist'}), train_ds)
    test_ds  = map(lambda x : XYDataset(x[0], x[1], **{'source': 'mnist'}), test_ds)

    return train_ds, test_ds


def get_split_cifar10(args):
    # assert args.n_tasks in [5, 10], 'SplitCifar only works with 5 or 10 tasks'
    assert '1.' in str(torch.__version__)[:2], 'Use Pytorch 1.x!'
    args.n_tasks   = 5
    args.n_classes = 10
    args.multiple_heads = False
    args.n_classes_per_task = 2
    args.input_size = (3, 32, 32)

    # fetch MNIST
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

        train_size = tr_e - tr_s
        split = tr_s + int(0.9 * train_size.item())

        train_ds += [(train_x[tr_s:split], train_y[tr_s:split])]
        valid_ds += [(train_x[split:tr_e], train_y[split:tr_e])]
        test_ds  += [(test_x[te_s:te_e],  test_y[te_s:te_e])]

    train_ds  = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), train_ds)
    valid_ds  = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), valid_ds)
    test_ds   = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), test_ds)

    return train_ds, valid_ds, test_ds


def get_split_cifar100(args):
    # assert args.n_tasks in [5, 10], 'SplitCifar only works with 5 or 10 tasks'
    assert '1.' in str(torch.__version__)[:2], 'Use Pytorch 1.x!'
    args.n_tasks   = 20
    args.n_classes = 100
    args.multiple_heads = True
    args.n_classes_per_task = 5
    args.input_size = (3, 32, 32)

    # fetch MNIST
    train = datasets.CIFAR100('../cl-pytorch/data/', train=True,  download=True)
    test  = datasets.CIFAR100('../cl-pytorch/data/', train=False, download=True)

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

    # TODO: check this
    # shuffle(indices)

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
    ROOT_PATH = '/mnt/data/lpagec/imagenet/imagenet_images'
    ROOT_PATH_CSV = '../prototypical-network-pytorch/materials'

    size = args.data_size
    args.n_tasks   = 20
    args.n_classes = 100
    args.multiple_heads = True
    args.n_classes_per_task = 5
    args.input_size = (3, size, size)

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
