import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from os.path import join
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from copy   import deepcopy
from pydoc  import locate

from buffer  import *
from utils   import *
from data    import *
from args    import get_args
from modular import QStack

args = get_args()

# functions
Mean = lambda x : sum(x) / len(x)
rescale_inv = (lambda x : x * 0.5 + 0.5)

# spawn writer
log_dir    = join('runs', args.model_name)
sample_dir = join(log_dir, 'samples')
writer     = SummaryWriter(log_dir=log_dir)

print(args)
best_test = float('inf')
maybe_create_dir(sample_dir)
print('logging into %s' % log_dir)
writer.add_text('hyperparameters', str(args), 0)

def eval(name, max_task=-1):
    """ evaluate performance on held-out data """

    with torch.no_grad():
        generator.eval()
        loader = valid_loader if 'valid' in name else test_loader

        for task_t, te_loader in enumerate(loader):
            # only eval on seen tasks
            if task_t > max_task >= 0:
                break

            for data, target in te_loader:
                data, target = data.to(args.device), target.to(args.device)
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
                from kitti_utils import from_polar
                all_samples = (all_samples if args.xyz else from_polar(all_samples))[:12]
                np.save(open('lidars/{}_test{}_{}'.format(args.model_name, task_t, max_task), 'wb'),
                        all_samples.cpu().data.numpy(), allow_pickle=False)

            else:
                save_image(rescale_inv(all_samples).view(-1, *args.input_size), \
                    'samples/{}_test_{}_{}.png'.format(args.model_name, task_t, max_task))


def eval_drift(real_img, indices):
    """ evaluate how much the compressed representations deviate from ground truth """

    assert len(real_img) == len(indices) # same amt of tasks

    with torch.no_grad():
        generator.eval()

        for task_t, (real_data, idx) in enumerate(zip(real_img, indices)):

            # 1) decode from old indices
            old_recons = generator.decode_indices(idx)

            # 2) push the real data through the model to fetch the new argmin indices
            out = generator(real_data)
            new_idx = generator.fetch_indices()

            # recons and indices are ordered from block T, block T - 1, ... block 1
            # so we iterate over the blocks in the same order

            for i, block in enumerate(reversed(generator.blocks)):

                # log the reconstruction error for every block
                # loss_i = F.mse_loss(old_recons[i], real_data)
                loss_i = F.mse_loss(old_recons[i], real_data)
                block.log.log('B%d-Full_recon-drift' % block.id, loss_i)

                # log the relative change in argmin indices
                cnt, total = 0., 0.
                for old_idx_ij, new_idx_ij in zip(idx[i], new_idx[i]):
                    cnt   += (old_idx_ij != new_idx_ij).sum().item()
                    total += old_idx_ij.numel()

                block.log.log('B%d-idx-drift' % block.id, cnt / total)

            generator.log(task_t, writer=writer, should_print=True, mode='buffer')



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

for run in range(1): #args.n_runs):

    # reproducibility
    set_seed(args.seed)

    # fetch data
    data = locate('data.get_%s' % args.dataset)(args)

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

            for i, (input_x, input_y) in enumerate(tr_loader):
                if i % 5 == 0 : print('  ', i, ' / ', len(tr_loader) , end='\r')
                sample_amt += input_x.size(0)

                input_x, input_y = input_x.to(args.device), input_y.to(args.device)
                if sample_amt > args.samples_per_task > 0: break

                if task > 0:
                    generator.update_old_decoder()

                for n_iter in range(args.n_iters):

                    if task > 0 and args.rehearsal:
                        re_x, re_y, re_t = generator.sample_from_buffer(input_x.size(0))
                        data_x, data_y = torch.cat((input_x, re_x)), torch.cat((input_y, re_y))
                    else:
                        data_x, data_y = input_x, input_y

                    out = generator(data_x,    **kwargs)
                    generator.optimize(data_x, **kwargs)

                    if task > 0 and args.rehearsal:
                        # potentially update the indices of `re_x`, or even it's compression level
                        generator.buffer_update_idx(re_x, re_y, re_t)

                if (i + 1) % 40 == 0 or (i+1) == len(tr_loader):
                    generator.log(task, writer=writer, should_print=True, mode='train')

                if args.rehearsal:
                    generator.add_reservoir(input_x, input_y, task)

        generator.eval()
        generator(input_x)
        to_be_added = (input_x, generator.fetch_indices())

        buffer_sample = generator.sample_from_buffer(64)[0]
        save_image(rescale_inv(buffer_sample), 'samples/%s_buffer_%d.png' % (args.model_name, task), nrow=8)

        if task > 0:
            if args.update_representations:

                # decode using the old generator, and encode back with new ones
                new_indices = []
                for idx in drift_indices:
                    # TODO: support multi level decoding & re-encoding
                    old_recon = prev_gen.decode_indices(idx)[-1]

                    generator(old_recon)
                    new_indices += [generator.fetch_indices()]

                drift_indices = new_indices

            # TODO: put this back
            #eval_drift(drift_images, drift_indices)

        # store the last minibatch
        drift_images  += [to_be_added[0]]
        drift_indices += [to_be_added[1]]

        if args.rehearsal or args.update_representations:
            argtemp = deepcopy(args)
            argtemp.rehearsal=False
            for kk,bloc in enumerate(generator.blocks):
                argtemp.layers[kk].rehearsal = False
            prev_gen = QStack(argtemp).to(args.device)
            prev_gen.load_state_dict(deepcopy(generator.state_dict()),strict=False)

        # evaluate on valid set
        eval('valid', max_task=task)
        generator.train()
        print('\n\n')
