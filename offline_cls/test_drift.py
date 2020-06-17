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

args = get_args()
print(args)

Mean = lambda x : sum(x) / len(x)
rescale_inv = (lambda x : x * 0.5 + 0.5)

# spawn writer
args.log_dir = join(args.run_dir, args.model_name)
sample_dir   = join(args.log_dir, 'samples')
writer       = SummaryWriter(log_dir=join(args.log_dir, 'tf'))

print_and_save_args(args, args.log_dir)
print('logging into %s' % args.log_dir)
maybe_create_dir(sample_dir)
best_test = float('inf')

def sho(x, nrow=8):
    save_image(x * .5 + .5, 'tmp.png', nrow=nrow)
    Image.open('tmp.png').show()

FINAL = []

def eval_drift(generator, max_task=-1, prefix=''):
    print('B0 %d' % generator.blocks[0].n_samples)
    with torch.no_grad():
        '''
        TODO:
            1) iterate over all the data in the buffer. Proceed level by level (no need for task by task for now)
            2) keep a drift measure for every level.
        '''

        generator.eval()
        gen_iter = generator.sample_EVERYTHING()

        all_loaders = list(valid_loader)

        cached = False
        for i, batch in enumerate(gen_iter):

            x_t, y_t, argmin_t, task_t, idx_t, block_id = batch

            if block_id == -1: continue

            if not cached:
                cached = True
                global FINAL
                FINAL += [x_t]

            target = []
            for _idx, _task in zip(idx_t, task_t):
                loader  = all_loaders[_task]
                target += [loader.dataset.__getitem__(_idx.item())[0]]

            if 'kitti' in args.dataset:
                target = torch.from_numpy(np.stack(target))
            else:
                target = torch.stack(target).to(x_t.device)

            target = target.to(x_t.device)

            generator.blocks[0].log.log('drift_mse_%d' % block_id, F.mse_loss(x_t, target), per_task=False)
            generator.blocks[0].log.log('drift_mse_total', F.mse_loss(x_t, target), per_task=False)

        generator.log(task, writer=writer, mode='buffer', should_print=True)


# ------------------------------------------------------------------------------
# Train the model
# ------------------------------------------------------------------------------

data = locate('utils.data.get_%s' % args.dataset)(args)

for run in range(args.n_runs):

    # reproducibility
    set_seed(args.seed)

    # fetch data
    data = locate('utils.data.get_%s' % args.dataset)(args)

    # make dataloaders
    train_loader, valid_loader, test_loader  = [
            CLDataLoader(elem, args, train=t) for elem, t in \
                    zip(data, [True, False, False])]
    vl_loader = list(valid_loader)[0]

    kwargs = {'all_levels_recon':True}

    # fetch model and ship to GPU
    generator  = QStack(args).to(args.device)
    print(generator)

    print("number of generator  parameters:", \
            sum([np.prod(p.size()) for p in generator.parameters()]))

    for task, tr_loader in enumerate(train_loader):
        print('\ntask %d' % task)

        for epoch in range(args.num_epochs):
            generator.train()


            for i, (input_x, input_y, idx_) in enumerate(tr_loader):
                if i % 5 == 0 : print('  ', i, ' / ', len(tr_loader), end='\r')

                generator.blocks[0].log.log('task', task, per_task=False)

                input_x = input_x.to(args.device)
                input_y = input_y.to(args.device)
                idx_    = idx_.to(args.device)

                for n_iter in range(args.n_iters):
                    if generator.all_stored > 0:
                        re_x, re_y, re_t, re_idx, _ = \
                                    generator.sample_from_buffer(input_x.size(0))

                        data_x = torch.cat((input_x, re_x))
                        data_y = torch.cat((input_y, re_y))
                    else:
                        data_x, data_y = input_x, input_y

                    out = generator(data_x,    **kwargs)

                    ''' Optimization '''
                    generator.optimize(data_x, **kwargs)

                    # Log the stuff that would usually be done in add_to_buffer
                    with torch.no_grad():

                        # first we add all incoming points
                        target = generator.all_levels_recon[:, :input_x.size(0)]

                        # now that we know which samples will be added to the buffer,
                        # we need to find the most compressed representation that is good enough

                        per_block_l2 = (input_x.unsqueeze(0) - target).pow(2)
                        per_block_l2 = per_block_l2.mean(dim=(2,3,4))
                        generator.avg_l2  = generator.avg_l2   * 0.99 + .01 * per_block_l2.mean(-1).flip(0)

                        recon_th  = generator.recon_th.unsqueeze(1).expand_as(per_block_l2)
                        block_id  = (per_block_l2 < recon_th)
                        comp_rate = block_id.float().mean(dim=1).flip(0)

                        for i, block in enumerate(generator.blocks):
                            # log
                            block.avg_comp = block.avg_comp * 0.99 + .01 * comp_rate[i].item()
                            block.log.log('buffer-%d-comp_rate' % block.id, block.avg_comp, per_task=False)
                            block.log.log('buffer-%d-avg_l2'    % block.id, generator.avg_l2[i].item(), per_task=False)

                if generator.blocks[0].avg_comp > .9 and generator.all_stored == 0:
                    generator.eval()
                    # add images from the validation set to the buffer
                    for i, (input_x, input_y, idx_) in enumerate(vl_loader):
                        input_x = input_x.to(args.device)
                        input_y = input_y.to(args.device)
                        idx_    = idx_.to(args.device)

                        out = generator(input_x, **kwargs)
                        print(F.mse_loss(out, input_x))

                        # generator.add_reservoir(input_x, input_y, task, idx_)
                        generator.add_reservoir(input_x, input_y, 0, idx_)

                    generator.train()

                    # evaluate drift at t == 0
                    eval_drift(generator)


                # add compressed rep. to buffer (ONLY during last epoch)
                if (i+1) % 20 == 0:
                    generator.log(task, writer=writer, mode='train', should_print=True)

        # evaluate drift
        eval_drift(generator)

        # print training stats
        generator.log(task, writer=writer, mode='train', should_print=True)

# out = rescale_inv(torch.cat(FINAL))
# save_image(out, 'extract_drift/FREEZE:%d.png' % args.freeze_embeddings, nrow=FINAL[0].size(0), padding=0)

# save model
if not args.debug:
    save_path = join(args.log_dir, 'gen.pth')
    print('saving model to %s' % save_path)
    torch.save(generator.state_dict(), save_path)

