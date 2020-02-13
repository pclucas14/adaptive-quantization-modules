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
import utils
from utils.data   import *
from utils.buffer import *
from utils.utils  import *
from utils.args   import get_args

from utils.kitti_utils import show_pc, from_polar

from common.modular import QStack

SAVE_TARGET = False

gen_weights = None

base_dir =  '/home/ml/lpagec/pytorch/stacked-quantized-modules/lidar/new_wave/'
run_id   = sys.argv[1]

for run in os.listdir(base_dir):
    if run_id in run:
        print('found run : %s' % run)
        gen_weights = os.path.join(base_dir, run)
        break

if gen_weights is None:
    import pdb; pdb.set_trace()


generator, args = load_model_from_file(gen_weights)
generator   = generator.to(args.device)

if SAVE_TARGET:
    data = utils.data.get_kitti(args)
    # make dataloaders
    train_loader = CLDataLoader(data[0], args, train=True)


def dump(lidar):
    np.save(open('../lidars/tmp', 'wb'),
        lidar.cpu().data.numpy(), allow_pickle=False)



with torch.no_grad():
    '''
    TODO:
        1) iterate over all the data in the buffer. Proceed level by level (no need for task by task for now)
        2) keep a drift measure for every level.
    '''

    generator.eval()
    gen_iter = generator.sample_EVERYTHING()

    i = 0
    for batch in gen_iter:
        print(i)

        x_t, y_t, _, task_t, idx_t, block_id = batch

        if block_id == -1: continue

        import pdb; pdb.set_trace()


        # save buffered sample
        all_samples = x_t

        all_samples = (all_samples if args.xyz else from_polar(all_samples))
        np.save(open('../lidars/buffer_%d' % i, 'wb'),
                all_samples.cpu().data.numpy(), allow_pickle=False)


        if SAVE_TARGET:
            # build target
            target = []
            for _idx, _task in zip(idx_t, task_t):
                loader = None
                for task__, loader_ in enumerate(train_loader):
                    loader = loader_
                    if task__ == _task: break

                target += [loader.dataset.__getitem__(_idx.item())[0]]

            target = torch.from_numpy(np.stack(target)).to(x_t.device)

            np.save(open('../lidars/target_%d' % i, 'wb'),
                    target.cpu().data.numpy(), allow_pickle=False)


        i += 1



