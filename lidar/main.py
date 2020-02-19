import sys
import torch
import numpy as np
from os.path import join
from pydoc  import locate
from copy   import deepcopy
import torch.nn.functional as F
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# sys.path.append('../../chamferdist')
# from chamferdist import ChamferDistance

sys.path.append('../')
sys.path.append('./chamfer_distance')
from chamfer_distance.chamfer_distance import ChamferDistance

from utils.data   import *
from utils.buffer import *
from utils.utils  import *
from utils.args   import get_args

from utils.kitti_utils import show_pc, from_polar

from common.modular import QStack

args = get_args()
args.normalize = False
print(args)

# functions
chamfer_raw = ChamferDistance()
Mean = lambda x : sum(x) / len(x)
rescale_inv = (lambda x : x * 0.5 + 0.5)
prepro = lambda x : x.reshape(x.size(0), 3, -1).transpose(-2, -1)
chamfer = lambda x, y : chamfer_raw(prepro(from_polar(x)), prepro(from_polar(y)))[:2]

# spawn writer
args.log_dir = join('icml_fix', args.model_name) + args.suffix
sample_dir   = join(args.log_dir, 'samples')
writer       = SummaryWriter(log_dir=args.log_dir)

def dump(lidar, nn='tmp'):
    np.save(open('lidars/%s' % nn, 'wb'),
        lidar.cpu().data.numpy(), allow_pickle=False)

print(args)
best_test = float('inf')
maybe_create_dir(sample_dir)
maybe_create_dir('lidars')
print_and_save_args(args, args.log_dir)
print('logging into %s' % args.log_dir)
writer.add_text('hyperparameters', str(args), 0)

def eval(name, max_task=-1, break_after=-1):
    """ evaluate performance on held-out data """

    with torch.no_grad():
        generator.eval()
        loader = valid_loader if 'valid' in name else test_loader

        for task_t, te_loader in enumerate(loader):
            # only eval on seen tasks
            if task_t > max_task >= 0:
                break

            # if task_t not in [0, max_task]: continue

            for i_, (data, target, _) in enumerate(te_loader):
                data, target = data.to(args.device), target.to(args.device)
                if i_ > break_after > 0: break

                # normalize point cloud
                max_ = data.reshape(data.size(0), -1).abs().max(dim=1)[0].view(-1, 1, 1, 1)
                data = data / max_

                # block_n, block_n-1, ..., block_0
                outs  = generator.reconstruct_all_levels(data)

                for i, out in enumerate(outs):
                    block_id = len(generator.blocks) - i
                    dist_a, dist_b  = chamfer(data * max_, out * max_)
                    snnrmse = (.5 * dist_a.mean(-1) + .5 * dist_b.mean(-1)).sqrt().mean()

                    generator.blocks[0].log.log('snnrmse_%d' % block_id, snnrmse, per_task=False)

            generator.log(task_t, writer=writer, should_print=True, mode=name)

        if max_task >= 0:
            outs += [data]
            all_samples = torch.stack(outs)             # L, bs, 3, 32, 32
            all_samples = all_samples.transpose(1,0)
            all_samples = all_samples.contiguous()      # bs, L, 3, 32, 32
            all_samples = all_samples.view(-1, *data.shape[-3:])

            all_samples = (all_samples if args.xyz else from_polar(all_samples))[:12]
            np.save(open('lidars/{}_test{}_{}'.format(args.model_name, task_t, max_task), 'wb'),
                    all_samples.cpu().data.numpy(), allow_pickle=False)


def eval_drift(max_task=-1):
    with torch.no_grad():
        '''
        TODO:
            1) iterate over all the data in the buffer. Proceed level by level (no need for task by task for now)
            2) keep a drift measure for every level.
        '''

        generator.eval()
        gen_iter = generator.sample_EVERYTHING()

        for batch in gen_iter:
            x_t, y_t, _, task_t, idx_t, block_id = batch

            if block_id == -1: continue

            # collect targets
            target = []
            for _idx, _task in zip(idx_t, task_t):
                loader = None
                for task__, loader_ in enumerate(train_loader):
                    loader = loader_
                    if task__ == _task: break
                try:
                    target += [loader.dataset.__getitem__(_idx.item())[0]]
                except:
                    import pdb; pdb.set_trace()
                    xx = 1

            target = torch.from_numpy(np.stack(target))
            target = target.to(x_t.device)

            # calculate MSE with normalized target
            max_ = target.reshape(x_t.size(0), -1).abs().max(dim=1)[0].view(-1, 1, 1, 1)
            target = target / max_

            diff = (x_t - target).pow(2).mean(dim=(1,2,3))
            diff = diff[diff != 0.].mean()

            dist_a, dist_b  = chamfer(target * max_, x_t * max_)
            snnrmse = (.5 * dist_a.mean(-1) + .5 * dist_b.mean(-1)).sqrt().mean()
            generator.blocks[0].log.log('drift_snnrmse_%d' % block_id, snnrmse, per_task=False)

            # remove nan
            if diff != diff: diff = torch.Tensor([0.])

            #mses += [diff.item()]
            generator.blocks[0].log.log('drift_mse_%d' % block_id, F.mse_loss(x_t, target), per_task=False)
            generator.blocks[0].log.log('drift_mse_total', F.mse_loss(x_t, target), per_task=False)

        generator.log(task, writer=writer, mode='buffer', should_print=True)


# -------------------------------------------------------------------------------
# Train the model
# -------------------------------------------------------------------------------

""" define training args """
kwargs = {'all_levels_recon':True}

for run in range(args.n_runs):

    # reproducibility
    set_seed(args.seed)

    # fetch data
    import time
    ss = time.time()
    data = locate('utils.data.get_%s' % args.dataset)(args)
    print('data prepro took {:.4f} seconds'.format(time.time() - ss))

    # build buffers to store data & indices (for drift monitoring)
    drift_images, drift_indices = [], []

    # make dataloaders
    train_loader, valid_loader, test_loader  = \
            [CLDataLoader(elem, args, train=t) for elem, t in zip(data, [True] + [False] * 2)]

    # fetch model and ship to GPU
    generator  = QStack(args).to(args.device)

    print("number of generator  parameters:", sum([np.prod(p.size()) for p \
            in generator.parameters()]))

    step = 0
    for task, tr_loader in enumerate(train_loader):
        print('task %d' % task)

        for epoch in range(args.num_epochs):
            generator.train()
            sample_amt = 0

            for i, (input_x, input_y, idx_) in enumerate(tr_loader):

                # normalize point cloud
                input_x, input_y, idx_ = input_x.to(args.device), input_y.to(args.device), idx_.to(args.device)
                max_ = input_x.reshape(input_x.size(0), -1).abs().max(dim=1)[0].view(-1, 1, 1, 1)
                input_x = input_x / max_

                if i % 5 == 0 : print('  ', i, ' / ', len(tr_loader) , end='\r')
                sample_amt += input_x.size(0)

                if sample_amt > args.samples_per_task > 0: break

                for n_iter in range(1 if generator.blocks[0].frozen_qt else args.n_iters):

                    if task > 0 and args.rehearsal:
                        re_x, re_y, re_t, re_idx, re_step = generator.sample_from_buffer(args.buffer_batch_size)
                        data_x, data_y = torch.cat((input_x, re_x)), torch.cat((input_y, re_y))
                    else:
                        data_x, data_y = input_x, input_y

                    out = generator(data_x,    **kwargs)
                    generator.optimize(data_x, **kwargs)

                    dist_a, dist_b  = chamfer(input_x* max_, out[:input_x.size(0)] * max_)
                    snnrmse = (.5 * dist_a.mean(-1) + .5 * dist_b.mean(-1)).sqrt().mean()
                    generator.blocks[0].log.log('inc_snnrmse_0', snnrmse, per_task=False)

                    if task > 0 and args.rehearsal:
                        # potentially update the indices of `re_x`, or even it's compression level
                        generator.buffer_update_idx(re_x, re_y, re_t, re_idx, re_step)

                    if args.rehearsal and n_iter == 0:
                        #generator.add_reservoir(input_x, input_y, task, idx_, step=step)
                        generator.add_to_buffer(input_x, input_y, task, idx_, step=step)

                if (i + 1) % 40 == 0 or (i+1) == len(tr_loader):
                    generator.log(task, writer=writer, should_print=True, mode='train')

                generator.update_ema_decoder()
                step += 1

            # Test the model
            # -------------------------------------------------------------------------------
            eval_drift(max_task=task)

            # save buffer samples
            buffer_sample = torch.stack((out, data_x)).transpose(1,0).reshape(out.size(0) + data_x.size(0), *out.shape[1:])

            from utils.kitti_utils import from_polar
            buffer_sample = buffer_sample[torch.randperm(buffer_sample.size(0))][:12]
            buffer_sample = (buffer_sample if args.xyz else from_polar(buffer_sample))[:12]
            np.save(open('lidars/{}_buf{}'.format(args.model_name, task), 'wb'),
                    buffer_sample.cpu().data.numpy(), allow_pickle=False)

            # save reconstructions
            outs = [input_x, out[:input_x.size(0)]]
            all_samples = torch.stack(outs)             # L, bs, 3, 32, 32
            all_samples = all_samples.transpose(1,0)
            all_samples = all_samples.contiguous()      # bs, L, 3, 32, 32
            all_samples = all_samples.view(-1, *input_x.shape[-3:])

            all_samples = (all_samples if args.xyz else from_polar(all_samples))[:12]
            np.save(open('lidars/{}_incoming_{}'.format(args.model_name, task), 'wb'),
                    all_samples.cpu().data.numpy(), allow_pickle=False)


    # save model
    if not args.debug:
        save_path = join(args.log_dir, 'gen.pth')
        print('saving model to %s' % save_path)
        torch.save(generator.state_dict(), save_path)
