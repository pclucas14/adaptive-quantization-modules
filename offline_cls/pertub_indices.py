import sys
import numpy as np
from os.path import join
from pydoc  import locate
from copy   import deepcopy
from collections import defaultdict
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torch.distributions import Categorical, RelaxedOneHotCategorical, Normal

sys.path.append('../')
from utils.data   import *
from utils.buffer import *
from utils.utils  import *
from utils.args   import get_args

from common.modular import QStack
from common.model   import ResNet18

np.set_printoptions(threshold=3)

Mean = lambda x : sum(x) / len(x)
rescale_inv = (lambda x : x * 0.5 + 0.5)

PATH = 'runs/DSniimagenet_NB1_RTH[0.025]_Comp24.00^_NI1_2091'
generator, args = load_model_from_file(PATH)

# spawn writer
args.log_dir = join(args.run_dir, 'test_pertub')
sample_dir   = join(args.log_dir, 'samples')
writer       = SummaryWriter(log_dir=join(args.log_dir, 'tf'))

def sho(x, nrow=8):
    save_image(x * .5 + .5, 'tmp.png', nrow=nrow)
    Image.open('tmp.png').show()

def eval_drift(generator, max_task=-1, prefix=''):
    print('B0 %d' % generator.blocks[0].n_samples)
    with torch.no_grad():
        '''
        TODO:
            1) iterate over all the data in the buffer. Proceed level by level (no need for task by task for now)
            2) keep a drift measure for every level.
        '''

        for noisy_p in [0, 0.01, 0.5, .9999]:
            kwargs = {'noisy_qt':noisy_p}

            generator.eval()
            gen_iter = generator.sample_EVERYTHING(**kwargs)

            all_loaders = list(valid_loader)

            for i, batch in enumerate(gen_iter):

                x_t, y_t, argmin_t, task_t, idx_t, block_id = batch

                if block_id == -1: continue

                target = []
                for _idx, _task in zip(idx_t, task_t):
                    loader  = all_loaders[_task]
                    target += [loader.dataset.__getitem__(_idx.item())[0]]

                target = torch.stack(target).to(x_t.device)
                target = target.to(x_t.device)

                generator.blocks[0].log.log('drift_mse_{:.4f}'.format(noisy_p), \
                        F.mse_loss(x_t, target), per_task=False)
                generator.blocks[0].log.log('drift_mse_total', F.mse_loss(x_t, target), per_task=False)

            generator.log(0, writer=writer, mode='buffer', should_print=True)


# ------------------------------------------------------------------------------
# Train the model
# ------------------------------------------------------------------------------

data = locate('utils.data.get_%s' % args.dataset)(args)

# make dataloaders
train_loader, valid_loader, test_loader  = [
        CLDataLoader(elem, args, train=t) for elem, t in \
                zip(data, [True, False, False])]
vl_loader = list(valid_loader)[0]

eval_drift(generator)
