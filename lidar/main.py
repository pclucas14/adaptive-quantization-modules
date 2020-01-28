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

from utils.kitti_utils import show_pc, from_polar

from common.modular import QStack

args = get_args()
args.normalize = True
print(args)

# functions
Mean = lambda x : sum(x) / len(x)
rescale_inv = (lambda x : x * 0.5 + 0.5)

# spawn writer
args.log_dir = join('new_wave', args.model_name)
sample_dir   = join(args.log_dir, 'samples')
writer       = SummaryWriter(log_dir=args.log_dir)

print(args)
best_test = float('inf')
maybe_create_dir(sample_dir)
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
                all_samples = (all_samples if args.xyz else from_polar(all_samples))[:12]
                np.save(open('../lidars/{}_test{}_{}'.format(args.model_name, task_t, max_task), 'wb'),
                        all_samples.cpu().data.numpy(), allow_pickle=False)

            else:
                save_image(rescale_inv(all_samples).view(-1, *args.input_size), \
                    '../samples/{}_test_{}_{}.png'.format(args.model_name, task_t, max_task))


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
            print(batch[0].shape, batch[-1])

            x_t, y_t, _, task_t, idx_t, block_id = batch

            if block_id == -1: continue

            if 'imagenet' in args.dataset or 'kitti' in args.dataset:
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

            #mses += [diff.item()]
            generator.blocks[0].log.log('drift_mse_%d' % block_id, F.mse_loss(x_t, target), per_task=False)
            generator.blocks[0].log.log('drift_mse_total', F.mse_loss(x_t, target), per_task=False)

        generator.log(task, writer=writer, mode='buffer', should_print=True)


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

        for epoch in range(args.num_epochs):
            generator.train()
            sample_amt = 0

            for i, (input_x, input_y, idx_) in enumerate(tr_loader):
                if i % 5 == 0 : print('  ', i, ' / ', len(tr_loader) , end='\r')
                sample_amt += input_x.size(0)

                input_x, input_y, idx_ = input_x.to(args.device), input_y.to(args.device), idx_.to(args.device)
                if sample_amt > args.samples_per_task > 0: break

                for n_iter in range(args.n_iters):

                    if task > 0 and args.rehearsal:
                        re_x, re_y, re_t, re_idx, re_step = generator.sample_from_buffer(args.buffer_batch_size)
                        data_x, data_y = torch.cat((input_x, re_x)), torch.cat((input_y, re_y))
                    else:
                        data_x, data_y = input_x, input_y

                    out = generator(data_x,    **kwargs)
                    generator.optimize(data_x, **kwargs)

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
            if task  % 5 == 0:
                eval_drift(max_task=task)
                eval('valid', max_task=task)

            buffer_sample, by, bt, _, _ = generator.sample_from_buffer(min(64, generator.all_stored - 5))
            if 'kitti' in args.dataset:
                from utils.kitti_utils import from_polar
                buffer_sample = buffer_sample[torch.randperm(buffer_sample.size(0))][:12]
                buffer_sample = (buffer_sample if args.xyz else from_polar(buffer_sample))[:12]
                np.save(open('../lidars/{}_buf{}'.format(args.model_name, task), 'wb'),
                        buffer_sample.cpu().data.numpy(), allow_pickle=False)
            else:
                save_image(rescale_inv(buffer_sample), '../samples/buf_%s_%d.png' % (args.model_name, task), nrow=8)

    # save model
    if not args.debug:
        save_path = join(args.log_dir, 'gen.pth')
        print('saving model to %s' % save_path)
        torch.save(generator.state_dict(), save_path)
