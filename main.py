import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from os.path import join
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
                outs  = generator.all_levels_recon(data)

            generator.log(prefix='%s task %d' % (name, task_t), writer=writer, should_print=True)

        if max_task >= 0:
            outs += [data]

            all_samples = torch.stack(outs)             # L, bs, 3, 32, 32
            all_samples = all_samples.transpose(1,0)
            all_samples = all_samples.contiguous()      # bs, L, 3, 32, 32
            all_samples = all_samples.view(-1, *data.shape[-3:])

            save_image(rescale_inv(all_samples).view(-1, *args.input_size), \
                    'samples/{}_test_{}_{}.png'.format(args.model_name, task_t, max_task))


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
    kwargs = {}

final_accs, finetune_accs  = [], []
for run in range(1): #args.n_runs):
    # reproducibility
    set_seed(521)

    # fetch data
    data = locate('data_.get_%s' % args.dataset)(args)

    # make dataloaders
    train_loader, valid_loader, test_loader  = [CLDataLoader(elem, args, train=t) for elem, t in zip(data, [True, False, False])]

    # TODO: fix this (so it generalizes to more than 1 layer)
    for i in range(len(args.layers.keys())):
        args.layers[i].in_channel = args.layers[i].out_channel = args.input_size[0] if i== 0 else args.layers[i].embed_dim
        args.layers[i].channel = args.layers[i].num_hiddens
        args.layers[i].stride = args.layers[i].downsample

    # fetch model and ship to GPU
    generator  = QStack(args).to(args.device)
    #print(generator)

    print("number of generator  parameters:", sum([np.prod(p.size()) for p \
            in generator.parameters()]))

    for task, tr_loader in enumerate(train_loader):

        for epoch in range(args.num_epochs):
            generator.train()
            sample_amt = 0

            for i, (input_x, input_y) in enumerate(tr_loader):
                if i % 5 == 0 : print('  ', i, ' / ', len(tr_loader), end='\r')

                if sample_amt > args.samples_per_task > 0: break
                sample_amt += input_x.size(0)

                input_x, input_y = input_x.to(args.device), input_y.to(args.device)

                for _ in range(args.n_iters):

                    # generator
                    generator(input_x, **kwargs)
                    generator.optimize(input_x, **kwargs)

                if (i + 1) % 30 == 0:
                    generator.log(prefix='Train task %d' % task, writer=writer, should_print=True)

                if (i + 1) % 120 == 0:
                    print('iteration %d ' % i)
                    eval('valid', max_task=task)
                    eval('test',  max_task=task)
                    generator.train()
                    print('\n\n')
