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

from common.modular import QStack

args = get_args()
print(args)

# functions
Mean = lambda x : sum(x) / len(x)
rescale_inv = (lambda x : x * 0.5 + 0.5)

# spawn writer
args.log_dir = join('runs_kitti_l1', args.model_name)
sample_dir   = join(args.log_dir, 'samples')
writer       = SummaryWriter(log_dir=args.log_dir)

print(args)
best_test = float('inf')
maybe_create_dir(sample_dir)
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

            for i_, (data, target, _) in enumerate(te_loader):
                data, target = data.to(args.device), target.to(args.device)
                if i_ > break_after > 0: break

                outs  = generator.reconstruct_all_levels(data)

            generator.log(task_t, writer=writer, should_print=True, mode=name)

        if max_task >= 0:
            outs += [data]
            all_samples = torch.stack(outs)             # L, bs, 3, 32, 32
            all_samples = all_samples.transpose(1,0)
            all_samples = all_samples.contiguous()      # bs, L, 3, 32, 32
            all_samples = all_samples.view(-1, *data.shape[-3:])

            if 'kitti' in args.dataset:
                # save lidar to samples
                from utils.kitti_utils import from_polar
                all_samples = (all_samples if args.xyz else from_polar(all_samples))[:12]
                np.save(open('../lidars/{}_test{}_{}'.format(args.model_name, task_t, max_task), 'wb'),
                        all_samples.cpu().data.numpy(), allow_pickle=False)

            else:
                save_image(rescale_inv(all_samples).view(-1, *args.input_size), \
                    '../samples/{}_test_{}_{}.png'.format(args.model_name, task_t, max_task))


def eval_drift(max_task=-1):
    with torch.no_grad():

        mses = []
        n_eval = min(1024, generator.all_stored - 10)

        if 'imagenet' in args.dataset or 'kitti' in args.dataset:
            n_eval = min(n_eval, 64)

        x, y, t, idx = generator.sample_from_buffer(n_eval)

        for task, loader in enumerate(train_loader):
            if task > max_task:
                break

            x_t   = x[t == task]
            y_t   = y[t == task]
            idx_t = idx[t == task]

            if idx_t.size(0) == 0:
                mses += [-1.]
                continue

            if 'imagenet' in args.dataset or 'kitti' in args.dataset:
                target = [loader.dataset.__getitem__(x.item())[0] for x in idx_t]
                if 'kitti' in args.dataset:
                    target = torch.from_numpy(np.stack(target))
                else:
                    target = torch.stack(target).to(x_t.device)
            elif 'cifar' in args.dataset:
                target = loader.dataset.rescale(loader.dataset.x[idx_t])

            target = target.to(x_t.device)
            diff = (x_t - target).pow(2).mean(dim=(1,2,3))
            diff = diff[diff != 0.].mean()

            # remove nan
            if diff != diff: diff = torch.Tensor([0.])

            mses += [diff.item()]
            generator.blocks[0].log.log('drift_mse', F.mse_loss(x_t, target))
            generator.log(task, writer=writer, mode='buffer', should_print=False)

        print('DRIFT : ', mses, '\n\n')


# -------------------------------------------------------------------------------
# Train the model
# -------------------------------------------------------------------------------

""" define training args """
if args.optimization == 'global':
    kwargs = {'inter_level_stream': True, 'inter_level_gradient':True}
    # --> does worse on all aspect, even learning the most compressed layer!
    # kwargs = {'inter_level_stream': False, 'inter_level_gradient':True}
    # --> learn the easiest layer a tiny bit faster, but the other ones don't learn
    #     (actually that was when recon was only on topmost (RGB)
    # --> actually it comes down to the same comp graph as blockwise opt.
else:
    kwargs = {'all_levels_recon':True}

for run in range(args.n_runs):

    # reproducibility
    set_seed(args.seed)

    # fetch data
    data = locate('utils.data.get_%s' % args.dataset)(args)

    # build buffers to store data & indices (for drift monitoring)
    drift_images, drift_indices = [], []

    # make dataloaders
    train_loader, valid_loader, test_loader  = \
            [CLDataLoader(elem, args, train=t) for elem, t in zip(data, [True] + [False] * 2)]

    # fetch model and ship to GPU
    generator  = QStack(args).to(args.device)

    print("number of generator  parameters:", sum([np.prod(p.size()) for p \
            in generator.parameters()]))

    for task, tr_loader in enumerate(train_loader):

        for epoch in range(args.num_epochs):
            generator.train()
            sample_amt = 0

            for i, (input_x, input_y, idx_) in enumerate(tr_loader):
                if i % 5 == 0 : print('  ', i, ' / ', len(tr_loader) , end='\r')
                sample_amt += input_x.size(0)

                input_x, input_y, idx_ = input_x.to(args.device), input_y.to(args.device), idx_.to(args.device)
                if sample_amt > args.samples_per_task > 0: break

                if task > 0:
                    generator.update_old_decoder()

                for n_iter in range(args.n_iters):

                    if task > 0 and args.rehearsal:
                        re_x, re_y, re_t, re_idx = generator.sample_from_buffer(input_x.size(0))
                        data_x, data_y = torch.cat((input_x, re_x)), torch.cat((input_y, re_y))
                    else:
                        data_x, data_y = input_x, input_y

                    out = generator(data_x,    **kwargs)
                    generator.optimize(data_x, **kwargs)

                    if task > 0 and args.rehearsal:
                        # potentially update the indices of `re_x`, or even it's compression level
                        generator.buffer_update_idx(re_x, re_y, re_t, re_idx)

                if (i + 1) % 40 == 0 or (i+1) == len(tr_loader):
                    generator.log(task, writer=writer, should_print=True, mode='train')

                if args.rehearsal:
                    generator.add_reservoir(input_x, input_y, task, idx_)

            # Test the model
            # -------------------------------------------------------------------------------
            generator.update_old_decoder()
            eval_drift(max_task=task)
            eval('valid', max_task=task)

            buffer_sample, by, bt, _ = generator.sample_from_buffer(64)
            if 'kitti' in args.dataset:
                from utils.kitti_utils import from_polar
                buffer_sample = buffer_sample[torch.randperm(64)][:12]
                buffer_sample = (buffer_sample if args.xyz else from_polar(buffer_sample))[:12]
                np.save(open('../lidars/{}_buf{}'.format(args.model_name, task), 'wb'),
                        buffer_sample.cpu().data.numpy(), allow_pickle=False)
            else:
                save_image(rescale_inv(buffer_sample), '../samples/buf_%s_%d.png' % (args.model_name, task), nrow=8)

    # save model
    save_path = join(args.log_dir, 'gen.pth')
    print('saving model to %s' % save_path)
    torch.save(generator.state_dict(), save_path)
