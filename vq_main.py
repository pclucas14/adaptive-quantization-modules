import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from os.path import join
from collections import OrderedDict as OD
from collections import defaultdict
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from copy   import deepcopy
from pydoc  import locate

from data   import *
from buffer import *
from utils  import *
from args    import get_args
from modular import QStack, ResNet18

import numpy as np
np.set_printoptions(threshold=3)

args = get_args()
print(args)

Mean = lambda x : sum(x) / len(x)
rescale_inv = (lambda x : x * 0.5 + 0.5)

# spawn writer
log_dir    = join('runs_cifar', args.model_name)
sample_dir = join(log_dir, 'samples')
writer     = SummaryWriter(log_dir=join(log_dir, 'tf'))

print_and_save_args(args, join(log_dir, 'args.txt'))
print('logging into %s' % log_dir)
maybe_create_dir(sample_dir)
best_test = float('inf')

def sho(x):
    save_image(x * .5 + .5, 'tmp.png')
    Image.open('tmp.png').show()


def eval_drift(max_task=-1):
    with torch.no_grad():

        mses = []
        n_eval = min(1024, generator.all_stored - 10)

        if 'imagenet' in args.dataset:
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

            if 'imagenet' in args.dataset:
                target = [loader.dataset.__getitem__(x.item())[0] for x in idx_t]
                target = torch.stack(target).to(x_t.device)
            else:
                target = loader.dataset.rescale(loader.dataset.x[idx_t]).to(x_t.device)

            diff = (x_t - target).pow(2).mean(dim=(1,2,3))
            diff = diff[diff != 0.].mean()

            # remove nan
            if diff != diff: diff = torch.Tensor([0.])

            mses += [diff.item()]
            generator.blocks[0].log.log('drift_mse', F.mse_loss(x_t, target))
            generator.log(task, writer=writer, mode='buffer', should_print=args.print_logs)

        print('DRIFT : ', mses, '\n\n')


def eval(name, max_task=-1):
    """ evaluate performance on held-out data """
    with torch.no_grad():
        generator.eval(); classifier.eval()

        logs = defaultdict(list)
        loader = valid_loader if 'valid' in name else test_loader

        for task_t, te_loader in enumerate(loader):
            # only eval on seen tasks
            if task_t > max_task >= 0:
                break

            for data, target, _ in te_loader:
                data, target = data.to(args.device), target.to(args.device)

                outs = generator.reconstruct_all_levels(data)
                out = generator(data,    **kwargs)
                logits = classifier(data)

                pred = logits.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().float()
                deno = data.size(0)

                logs['mse_%d'  % task_t]    += [F.mse_loss(out, data).item()]
                logs['acc_%d'  % task_t]    += [(correct / deno).item()]

            mean_acc = float(int(1000 * Mean(logs['acc_%d' % task_t]))) / 10.
            RESULTS[run][0 if name == 'valid' else 1][max_task][task_t] = mean_acc

            print('task %d' % task_t)
            generator.log(task_t, writer=writer, mode=name, should_print=args.print_logs)
            if max_task >= 0:
                outs += [data]
                all_samples = torch.stack(outs)             # L, bs, 3, 32, 32
                all_samples = all_samples.transpose(1,0)
                all_samples = all_samples.contiguous()      # bs, L, 3, 32, 32
                all_samples = all_samples.view(-1, *data.shape[-3:])

                save_image(rescale_inv(all_samples).view(-1, *args.input_size), \
                    'samples/{}_test_{}_{}.png'.format(args.model_name, task_t, max_task))

        """ Logging """
        logs = average_log(logs)
        print('Task {}, Epoch {}'.format(task, epoch))
        for name, value in logs.items():
            string = name + '\t:'
            for val in value.values():
                string += '{:.4f}\t'.format(val)

            string += '// {:.4f}'.format(Mean(value.values()))

            print(string)

        return logs



# -------------------------------------------------------------------------------
# Train the model
# -------------------------------------------------------------------------------

data = locate('data.get_%s' % args.dataset)(args)
RESULTS = np.zeros((args.n_runs, 2, args.n_tasks, args.n_tasks))

for run in range(args.n_runs):

    # reproducibility
    set_seed(run + 235)

    # fetch data
    data = locate('data.get_%s' % args.dataset)(args)
    RESULTS = np.zeros((args.n_runs, 2, args.n_tasks, args.n_tasks))

    # make dataloaders
    train_loader, valid_loader, test_loader  = [CLDataLoader(elem, args, train=t) for elem, t in zip(data, [True, False, False])]

    kwargs = {'all_levels_recon':True}

    # fetch model and ship to GPU
    generator  = QStack(args).to(args.device)
    classifier = ResNet18(args.n_classes, 20, input_size=args.input_size).to(args.device)

    print(generator)

    # optimizers
    opt_class = torch.optim.SGD(classifier.parameters(), lr=args.cls_lr)

    print("number of generator  parameters:", sum([np.prod(p.size()) for p in generator.parameters()]))
    print("number of classifier parameters:", sum([np.prod(p.size()) for p in classifier.parameters()]))

    for task, tr_loader in enumerate(train_loader):

        for epoch in range(1):
            generator.train()
            classifier.train()
            sample_amt = 0

            # create logging containers
            train_log = defaultdict(list)

            for i, (input_x, input_y, idx_) in enumerate(tr_loader):
                if i % 5 == 0 : print('  ', i, ' / ', len(tr_loader), end='\r')

                if sample_amt > args.samples_per_task > 0: break
                sample_amt += input_x.size(0)

                input_x, input_y, idx_ = input_x.to(args.device), input_y.to(args.device), idx_.to(args.device)

                for n_iter in range(args.n_iters):

                    if task > 0 and args.rehearsal:
                        re_x, re_y, re_t, re_idx = generator.sample_from_buffer(input_x.size(0) * 1)
                        data_x, data_y = torch.cat((input_x, re_x)), torch.cat((input_y, re_y))
                    else:
                        data_x, data_y = input_x, input_y

                    out = generator(data_x,    **kwargs)
                    generator.optimize(data_x, **kwargs)

                    if task > 0 and args.rehearsal:
                        # potentially update the indices of `re_x`, or even it's compression level
                        generator.buffer_update_idx(re_x, re_y, re_t, re_idx)

                    if n_iter < args.cls_n_iters:
                        opt_class.zero_grad()
                        logits = classifier(input_x)
                        loss_class = F.cross_entropy(logits, input_y)
                        loss_class.backward()
                        opt_class.step()

                        if task > 0:
                            opt_class.zero_grad()
                            logits = classifier(re_x)
                            loss_class = F.cross_entropy(logits, re_y)
                            loss_class.backward()
                            opt_class.step()

                generator.update_old_decoder()


                # add compressed rep. to buffer (ONLY during last epoch)
                if (i+1) % 200 == 0 or (i+1) == len(tr_loader):
                    generator.log(task, writer=writer, mode='train', should_print=args.print_logs)

                if args.rehearsal:
                    generator.add_reservoir(input_x, input_y, task, idx_)


            # Test the model
            # -------------------------------------------------------------------------------
            generator.update_old_decoder()
            eval_drift(max_task=task)
            eval('valid', max_task=task)
            eval('test', max_task=task)

        buffer_sample, by, bt, _ = generator.sample_from_buffer(64)
        save_image(rescale_inv(buffer_sample), 'samples/buffer_%d.png' % task, nrow=8)

np.save(join(log_dir, 'results'), RESULTS)

# Save final results
acc_avg = RESULTS.mean(axis=0)
acc_std = RESULTS.std(axis=0)

final_valid = acc_avg[0][-1]
final_test  = acc_avg[1][-1]

print('final valid:')
out = ''
for acc_, std_ in zip(acc_avg[0][-1], acc_std[0][-1]):
    out += '{:.2f} +- {:.2f}\t'.format(acc_, std_ * 2 / args.n_runs)
print(out)

print('{:.2f} +- {:.2f}'.format(RESULTS[:, 0, -1, :].mean(), RESULTS[:, 0, -1, :].std() * 2 / args.n_runs))

print('final test:')
out = ''
for acc_, std_ in zip(acc_avg[0][-1], acc_std[0][-1]):
    out += '{:.2f} +- {:.2f}\t'.format(acc_, std_ * 2 / args.n_runs)
print(out)

print('{:.2f} +- {:.2f}'.format(RESULTS[:, 1, -1, :].mean(), RESULTS[:, 1, -1, :].std() * 2 / args.n_runs))
