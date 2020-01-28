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
from utils.data   import *
from utils.buffer import *
from utils.utils  import *
from utils.args   import get_args

from utils.kitti_utils import show_pc, from_polar, get_chamfer_dist, from_polar

from common.modular import QStack

gen_weights = '/home/ml/lpagec/pytorch/stacked-quantized-modules/lidar/new_wave/DSkitti_NB2_RTH[8e-05, 8e-05]_Comp8.00^16.00^_Coef2.60_1690'
gen_weights = '/home/ml/lpagec/pytorch/stacked-quantized-modules/lidar/new_wave/DSkitti_NB2_RTH[8e-05, 8e-05]_Comp8.00^32.00^_Coef2.60_1736'
gen_weights = '/home/ml/lpagec/pytorch/stacked-quantized-modules/lidar/new_wave/DSkitti_NB1_RTH[8e-05]_Comp8.00^_Coef2.60_8311/' # .8172
gen_weights = '/home/ml/lpagec/pytorch/stacked-quantized-modules/lidar/new_wave/DSkitti_NB1_RTH[8e-05]_Comp8.00^_Coef2.60_5289/'
gen_weights = '/home/ml/lpagec/pytorch/stacked-quantized-modules/lidar/new_wave/DSkitti_NB2_RTH[8e-05, 8e-05]_Comp5.33^10.67^_Coef2.60_918/'
generator, args = load_model_from_file(gen_weights)
generator   = generator.to(args.device)

args.normalize = False


data = locate('utils.data.get_%s' % args.dataset)(args)
# make dataloaders
train_loader = CLDataLoader(data[0], args, train=True)

with torch.no_grad():

    generator.eval()
    gen_iter = generator.sample_EVERYTHING()

    out = []
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
        raw_x_t = from_polar(x_t) * max_.view(-1, 1, 1, 1)

        c_dist = get_chamfer_dist()
        dist = [c_dist(raw_x_t[[i]], target[[i]]) for i in range(x_t.size(0))]

        dist_a = torch.stack([x[0] for x in dist]).squeeze(1).mean(-1)
        dist_b = torch.stack([x[1] for x in dist]).squeeze(1).mean(-1)

        snnrmse = (.5 * dist_a + .5 * dist_b).sqrt().mean()

        out += [snnrmse]

    final = sum(out) / len(out)

    print('average snnrmse : {:.4f}'.format(final))

        # TODO: figure out how to calculate chamfer from here.
        # might need to uncomment the scaling


