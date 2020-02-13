import os
import sys
import torch
import numpy as np
from os.path import join
from pydoc  import locate
from copy   import deepcopy
import torch.nn.functional as F
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

sys.path.append('../')
sys.path.append('../../chamferdist')
from chamferdist import ChamferDistance

from utils.data   import *
from utils.buffer import *
from utils.utils  import *
from utils.args   import get_args

from utils.kitti_utils import show_pc, from_polar, get_chamfer_dist, from_polar

from common.modular import QStack

base_dir =  '/home/ml/lpagec/pytorch/stacked-quantized-modules/lidar/new_wave/'
run_id   = sys.argv[1]

# normal = 1936
# run_id = '1439'
gen_weights = None

def dump(lidar, nn='tmp'):
    np.save(open('../lidars/%s' % nn, 'wb'),
        lidar.cpu().data.numpy(), allow_pickle=False)


for run in os.listdir(base_dir):
    if run_id in run:
        print('found run : %s' % run)
        gen_weights = os.path.join(base_dir, run)
        break

if gen_weights is None:
    import pdb; pdb.set_trace()

generator, args = load_model_from_file(gen_weights)
generator   = generator.to(args.device)

# fetch the target point clouds unnormalized to properly measure diff.
args.normalize = False

# make dataloaders
data = locate('utils.data.get_%s' % args.dataset)(args)
train_loader = CLDataLoader(data[0], args, train=True)

with torch.no_grad():

    generator.eval()
    gen_iter = generator.sample_EVERYTHING()

    out, dists_a, dists_b = [], [], []
    for batch in gen_iter:

        x_t, y_t, _, task_t, idx_t, block_id = batch

        if block_id == -1: continue

        if 'imagenet' in args.dataset or 'kitti' in args.dataset:
            target = []
            for _idx, _task in zip(idx_t, task_t):
                loader = None
                for task__, loader_ in enumerate(train_loader):
                    loader = loader_
                    if task__ == _task: break

                target += [loader.dataset.__getitem__(_idx.item())[0]]

            target = torch.from_numpy(np.stack(target)).to(x_t.device)

        # unscale buffer sample
        target = from_polar(target)
        max_    = target.view(target.size(0), -1).max(-1)[0]
        raw_x_t = xx = from_polar(x_t) * max_.view(-1, 1, 1, 1)

        c_dist = get_chamfer_dist()
        # import pdb; pdb.set_trace()
        scaling = torch.stack([target[i].abs().max() / raw_x_t[i].abs().max() for i in range(x_t.size(0))])

        '''
        dist = []
        for i in range(x_t.size(0)):
            raw = raw_x_t[i]
            tgt = target[i]
            scale = tgt.reshape(3, -1).abs().max(1)[0] / raw.reshape(3, -1).abs().max(1)[0]
            dist += [c_dist((raw * scale.view(-1, 1, 1)).unsqueeze(0), tgt.unsqueeze(0))]

        dist = [c_dist(raw_x_t[[i]] * scaling[i], target[[i]]) for i in range(x_t.size(0))]
        dist_a = torch.stack([x[0] for x in dist]).squeeze(1)
        dist_b = torch.stack([x[1] for x in dist]).squeeze(1)

        snnrmse = (.5 * dist_a.mean(-1) + .5 * dist_b.mean(-1)).sqrt().mean()
        '''

        ''' new Chamfer '''
        prepro = lambda x : x.reshape(x.size(0), 3, -1).transpose(-2, -1)
        chamfer = ChamferDistance()

        dist_a, dist_b, _, _ = chamfer(prepro(raw_x_t) * scaling.view(-1, 1, 1), prepro(target))
        snnrmse = (.5 * dist_a.mean(-1) + .5 * dist_b.mean(-1)).sqrt().mean()

        dists_a += [dist_a.mean()]
        dists_b += [dist_b.mean()]
        out     += [snnrmse]

    final = sum(out) / len(out)
    dist_a = sum(dists_a) / len(dists_a)
    dist_b = sum(dists_b) / len(dists_b)

    print('average snnrmse : {:.4f}'.format(final))
    print('average dist a  : {:.4f}'.format(dist_a))
    print('average dist b  : {:.4f}'.format(dist_b))

