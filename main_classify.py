import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from os.path import join
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import numpy as np
from copy   import deepcopy
from pydoc  import locate

from buffer  import *
from utils   import *
from data    import *
from args    import get_args
from modular import QStack, ResNet18
import datetime
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
##################### Logs
time_stamp = str(datetime.datetime.now().isoformat())
name_log_txt = args.dataset+'_'+time_stamp + str(np.random.randint(0, 1000)) + args.name
name_log_txt=name_log_txt +'.log'
with open(name_log_txt, "a") as text_file:
    print(args, file=text_file)

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
def eval_classifier(all_models):
    # Test the model
    # -------------------------------------------------------------------------------
    avgs = []
    with torch.no_grad():
        # Init
        n_t = len(test_loader)
        accuracies = {}
        # iterate over samples from task
        loss_, correct, deno = {}, {}, {}
        for run in range(args.n_runs):
            accuracies[run] = {}  # this is going to be Tasks x Runs
            loss_[run], correct[run], deno[run] = {}, {}, {}
            for task_model, model in enumerate(all_models[run]):
                accuracies[run][task_model] = []
                correct[run][task_model] = [0. for _ in range(n_t)]
                deno[run][task_model] = [0. for _ in range(n_t)]

        for task, loader in enumerate(test_loader):
            # iterate over samples from task
            for i, (data, target) in enumerate(loader):
                data, target = data.to(args.device), target.to(args.device)
                for run in range(args.n_runs):
                    for task_model, model in enumerate(all_models[run]):
                        model = model.eval().to(args.device)
                        logits = model(data)
                        if args.multiple_heads:
                            logits = logits.masked_fill(loader.dataset.mask == 0, -1e9)
                        loss = F.cross_entropy(logits, target)
                        pred = logits.argmax(dim=1, keepdim=True)
                        correct[run][task_model][task] += pred.eq(target.view_as(pred)).sum().item()
                        deno[run][task_model][task] += data.size(0)  # pred.size(0)
                        model = model.cpu()

        for run in range(args.n_runs):
            for task_model, _ in enumerate(all_models[run]):
                for task in range(n_t):
                    accuracies[run][task_model] += [float(correct[run][task_model][task]) / deno[run][task_model][task]]
            out = ''
            for i, acc in enumerate(accuracies[run][task_model]):
                out += '{} : {:.2f}\t'.format(i, acc)
            print(out)
            avgs += [sum(accuracies[run][task_model]) / len(accuracies[run][task_model])]
            #  print('Avg {:.5f}'.format(avgs[-1]), '\n')
            with open(name_log_txt, "a") as text_file:
                print(out, file=text_file)

    # print('Max loss = {}. AVG over {} runs : {:.4f}'.format(args.max_loss, args.n_runs, sum(avgs) / len(avgs)))
    for task_model in range(len(all_models[0])):
        avgs = []
        for run in range(args.n_runs):
            avgs += [
                sum(accuracies[run][task_model][:task_model + 1]) / len(accuracies[run][task_model][:task_model + 1])]

        avg = np.array(avgs).mean()
        std = np.array(avgs).std()
        with open(name_log_txt, "a") as text_file:
            print('After Task {} Max loss = {}. AVG over {} runs : {:.4f} +- {:.4f}'
                  .format(task_model, args.max_loss, args.n_runs, avg, std * 2. / np.sqrt(args.n_runs)), file=text_file)

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

all_models = {}
for run in range(1): #args.n_runs):

    # reproducibility
    set_seed(args.seed)
    model = ResNet18(args.n_classes, nf=20, compressed=True).to(args.device)
    opt =  torch.optim.SGD(model.parameters(), lr=0.01)
    all_models[run] = []
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

                for _ in range(args.n_iters):

                    if task > 0 and args.rehearsal:
                        # TODO: make this layer agnostic
                        re_x, re_y     = generator.sample_from_buffer(input_x.size(0))
                        data_x, data_y = torch.cat((input_x, re_x)), torch.cat((input_y, re_y))
                    else:
                        data_x, data_y = input_x, input_y

                    out = generator(data_x,    **kwargs)
                    generator.optimize(data_x, **kwargs)

                    #classifier
                    opt.zero_grad()
                    F.cross_entropy(model(data_x),data_y).backward()
                    opt.step()

                if (i + 1) % 60 == 0 or (i+1) == len(tr_loader):
                    generator.log(task, writer=writer, should_print=True, mode='train')

                if args.rehearsal:
                    generator.add_reservoir(input_x, input_y, task)
        all_models[run] += [deepcopy(model).cpu()]
        generator.eval()
        generator(input_x)
        to_be_added = (input_x, generator.fetch_indices())

        buffer_sample = generator.sample_from_buffer(64)[0]
        save_image(rescale_inv(buffer_sample), 'samples/buffer_%d.png' % task, nrow=8)

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
            eval_drift(drift_images, drift_indices)

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
        print(task)
        eval('valid', max_task=task)
        generator.train()
        print('\n\n')
    eval_classifier(all_models)
