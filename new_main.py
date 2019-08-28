import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from os.path import join
from collections import OrderedDict as OD
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from copy   import deepcopy
from pydoc  import locate

from utils   import *
from data_   import *
from args    import get_args
from modular import QStack

args = get_args()

# functions
Mean = lambda x : sum(x) / len(x)
rescale_inv = lambda x : x * 0.5 + 0.5

# spawn writer
#model_name = '{}_NI_{}_LR{}_BS{}_BBS{}_D{}_K{}_hH{}_coef{}'.format(('CIF' if 'cifar' in args.dataset else 'IMNET'), args.n_iters, args.lr, args.batch_size,
#                                                                    args.buffer_batch_size, args.hidden, args.k, args.hH, args.commit_coef)
model_name = 'test' #if args.debug else model_name
log_dir    = join('runs', model_name)
sample_dir = join(log_dir, 'samples')
writer     = SummaryWriter(log_dir=log_dir)

print(args)
print('logging into %s' % log_dir)
maybe_create_dir(sample_dir)
best_test = float('inf')

def eval(loader, max_task=-1):
    """ evaluate performance on held-out data """
    with torch.no_grad():
        generator.eval()

        logs = DefaultOrderedDict(list)
        loader = valid_loader if 'valid' in loader else test_loader

        pad = lambda x, y : x + ' ' * max(0, y - len(x))

        for task_t, te_loader in enumerate(loader):
            # only eval on seen tasks
            if task_t > max_task >= 0:
                break

            for data, target in te_loader:
                data, target = data.to(args.device), target.to(args.device)

                outs  = generator.all_levels_recon(data)
                logs['mse-in-0_%d' % task_t] += [F.mse_loss(outs[0], data).item()]
                logs['mse-in-1_%d' % task_t] += [F.mse_loss(outs[1], data).item()]

                for i, block in enumerate(generator.blocks):
                    for j, ppl in enumerate(block.ppls):
                        logs['ppl-%d-%d_%d' % (i,j, task_t)] += [ppl]
                    for j, diff in enumerate(block.diffs):
                        logs['diffs-%d-%d_%d' % (i,j, task_t)] += [diff]

        if max_task >= 0:
            print('Test Task {}'.format(max_task))
            outs  = generator.all_levels_recon(data)
            outs += [data]

            all_samples = torch.stack(outs)             # L, bs, 3, 32, 32
            all_samples = all_samples.transpose(1,0)
            all_samples = all_samples.contiguous()      # bs, L, 3, 32, 32
            all_samples = all_samples.view(-1, *data.shape[-3:])

            save_image(rescale_inv(all_samples).view(-1, *args.input_size), \
                    'samples/test_{}_{}.png'.format(task_t, max_task))

        """ Logging """
        logs = average_log(logs)
        print('Task {}, Epoch {}'.format(task, epoch))
        for name, value in logs.items():
            string = pad(name, 20) + '\t:'
            for val in value.values():
                string += '%#08.4f\t' % val

            string += '// %#08.4f' % Mean(value.values())

            print(string)

        return logs


# -------------------------------------------------------------------------------
# Train the model
# -------------------------------------------------------------------------------

""" define training args """
kwargs = {'inter_level_stream': True, 'inter_level_gradient':True}

final_accs, finetune_accs  = [], []
for run in range(1): #args.n_runs):
    # reproducibility
    set_seed(521)

    # fetch data
    data = locate('data_.get_%s' % args.dataset)(args)

    # make dataloaders
    train_loader, valid_loader, test_loader  = [CLDataLoader(elem, args, train=t) for elem, t in zip(data, [True, False, False])]

    prev_gen = None

    # TODO: fix this (so it generalizes to more than 1 layer)
    for i in range(len(args.layers.keys())):
        args.layers[i].in_channel = args.layers[i].out_channel = args.input_size[0] if i== 0 else args.layers[i].embed_dim
        args.layers[i].channel = args.layers[i].num_hiddens
        args.layers[i].stride = args.layers[i].downsample

    # fetch model and ship to GPU
    generator  = QStack(args).to(args.device)

    print(generator)

    print("number of generator  parameters:", sum([np.prod(p.size()) for p \
            in generator.parameters()]))

    opt = torch.optim.Adam(generator.parameters(), lr=1e-4)
    for task, tr_loader in enumerate(train_loader):

        for epoch in range(args.num_epochs):
            generator.train()
            sample_amt = 0

            # create logging containers
            train_log = reset_log()

            for i, (input_x, input_y) in enumerate(tr_loader):
                if i % 5 == 0 : print('  ', i, ' / ', len(tr_loader), end='\r')

                if sample_amt > args.samples_per_task > 0: break
                sample_amt += input_x.size(0)

                input_x, input_y = input_x.to(args.device), input_y.to(args.device)

                for _ in range(args.n_iters):

                    # generator
                    generator(input_x, **kwargs)
                    loss = generator.optimize(input_x, **kwargs)

                    #opt.zero_grad()
                    #loss.backward()
                    #opt.step()

                if (i + 1) % 120 == 0:
                    print('iteration %d ' % i)
                    eval('valid', max_task=task)
                    eval('test', max_task=task)
                    generator.train()
                    print('\n\n\n')



            # Test the model
            # -------------------------------------------------------------------------------
            #if True:#@(task + 1) % 5 == 0:
            #    eval('valid', max_task=task)
            #    eval('test', max_task=task)

