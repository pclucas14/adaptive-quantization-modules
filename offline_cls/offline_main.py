import sys
import numpy as np
from os.path import join
from pydoc  import locate
from copy   import deepcopy
from collections import defaultdict
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torchvision.utils import save_image

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
writer.add_text('hyperparameters', str(args), 0)

print_and_save_args(args, args.log_dir)
print('logging into %s' % args.log_dir)
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
            generator.log(task, writer=writer, mode='buffer', should_print=False)

        print('DRIFT : ', mses, '\n\n')


def eval_cls(name, max_task=-1, break_after=-1):
    """ evaluate performance on held-out data """
    with torch.no_grad():
        classifier.eval()

        logs = defaultdict(list)
        loader = valid_loader if 'valid' in name else test_loader

        for task_t, te_loader in enumerate(loader):
            # only eval on seen tasks
            if task_t > max_task >= 0:
                break

            for i_, (data, target, _) in enumerate(te_loader):
                data, target = data.to(args.device), target.to(args.device)
                if i_ > break_after > 0: break

                if args.test_on_recon:
                    logits = classifier(out)
                else:
                    logits = classifier(data)

                if args.multiple_heads:
                    logits = logits.masked_fill(te_loader.dataset.mask == 0, -1e9)

                pred = logits.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().float()
                deno = data.size(0)

                logs['acc_%d'  % task_t]    += [(correct / deno).item()]

            mean_acc = float(int(1000 * Mean(logs['acc_%d' % task_t]))) / 10.

            res_idx = 0 if name == 'valid' else 1

            RESULTS[run][res_idx][max_task][task_t] = mean_acc

        print(RESULTS[-1, res_idx, -1].mean())
        return RESULTS[-1, res_idx, -1].mean()

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


def eval_gen(name, max_task=-1, break_after=-1):
    """ evaluate performance on held-out data """
    with torch.no_grad():
        generator.eval()

        logs = defaultdict(list)
        loader = valid_loader if 'valid' in name else test_loader

        for task_t, te_loader in enumerate(loader):
            # only eval on seen tasks
            if task_t > max_task >= 0:
                break

            for i_, (data, target, _) in enumerate(te_loader):
                data, target = data.to(args.device), target.to(args.device)

                outs = generator.reconstruct_all_levels(data)
                out  = generator(data,    **kwargs)

                if i_ > break_after > 0: break

                logs['mse_%d'  % task_t]    += [F.mse_loss(out, data).item()]

            print('task %d' % task_t)
            generator.log(task_t, writer=writer, mode=name, should_print=args.print_logs)
            if max_task >= 0:
                outs += [data]
                all_samples = torch.stack(outs)             # L, bs, 3, 32, 32
                all_samples = all_samples.transpose(1,0)
                all_samples = all_samples.contiguous()      # bs, L, 3, 32, 32
                all_samples = all_samples.view(-1, *data.shape[-3:])

                save_image(rescale_inv(all_samples).view(-1, *args.input_size), \
                    '../samples/{}_test_{}_{}.png'.format(args.model_name, task_t, max_task))

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

for run in range(args.n_runs):

    set_seed(args.seed)


    kwargs = {'all_levels_recon':True}
    if args.optimization == 'global':
        # flow gradient through blocks when doing global optimization
        kwargs.update({'inter_level_gradient':True})

    # fetch model and ship to GPU
    generator  = QStack(args).to(args.device)

    try:
        generator, args = load_model_from_file(args.gen_weights)

        # fetch data
        data = locate('utils.data.get_%s' % args.dataset)(args)

        # make dataloaders
        train_loader, valid_loader, test_loader  = [CLDataLoader(elem, args, train=t) for elem, t in zip(data, [True, False, False])]

        task, epoch = -1, -1

    except Exception as e:
        print(e, '\n\n')

        print("number of generator  parameters:", sum([np.prod(p.size()) for p in generator.parameters()]))

        # fetch data
        data = locate('utils.data.get_%s' % args.dataset)(args)

        # make dataloaders
        train_loader, valid_loader, test_loader  = [CLDataLoader(elem, args, train=t) for elem, t in zip(data, [True, False, False])]

        for task, tr_loader in enumerate(train_loader):
            for epoch in range(1):
                generator.train()
                sample_amt = 0

                # create logging containers
                train_log = defaultdict(list)

                for i, (input_x, input_y, idx_) in enumerate(tr_loader):
                    if i % 5 == 0 : print('  ', i, ' / ', len(tr_loader), end='\r')

                    if sample_amt > args.samples_per_task > 0 : break
                    sample_amt += input_x.size(0)

                    input_x, input_y, idx_ = input_x.to(args.device), input_y.to(args.device), idx_.to(args.device)

                    for n_iter in range(args.n_iters):

                        if task > 0 and args.rehearsal:
                            re_x, re_y, re_t, re_idx = generator.sample_from_buffer(args.buffer_batch_size)
                            data_x, data_y = torch.cat((input_x, re_x)), torch.cat((input_y, re_y))
                        else:
                            data_x, data_y = input_x, input_y

                        out = generator(data_x,    **kwargs)
                        generator.optimize(data_x, **kwargs)

                        if task > 0 and args.rehearsal:
                            # potentially update the indices of `re_x`, or even it's compression level
                            generator.buffer_update_idx(re_x, re_y, re_t, re_idx)

                    if args.n_iters > 0: generator.update_old_decoder()

                    if (i+1) % 200 == 0 or (i+1) == len(tr_loader):
                        generator.log(task, writer=writer, mode='train', should_print=args.print_logs)

                    if args.rehearsal:
                        generator.add_reservoir(input_x, input_y, task, idx_)


                # Test the model
                # -------------------------------------------------------------------------------
                generator.update_old_decoder()
                eval_drift(max_task=task)
                if task < 2 or (task % 7 == 0): eval_gen('valid', max_task=task, break_after=2)

            buffer_sample, by, bt, _ = generator.sample_from_buffer(64)
            save_image(rescale_inv(buffer_sample), '../samples/buf_%s_%d.png' % (args.model_name, task), nrow=8)

        # save model
        save_path = join(args.log_dir, 'gen.pth')
        print('saving model to %s' % save_path)
        torch.save(generator.state_dict(), save_path)

generator.cuda()
generator.eval()

RESULTS = np.zeros((args.n_runs, 2, args.n_tasks, args.n_tasks))

# optimizers
classifier = ResNet18(args.n_classes, 20, input_size=args.input_size).to(args.device)
print("number of classifier parameters:", sum([np.prod(p.size()) for p in classifier.parameters()]))
print('cls learning rate {:.4f}'.format(args.cls_lr))
opt_class = torch.optim.SGD(classifier.parameters(), lr=args.cls_lr, momentum=0.9)

last_valid_acc = 0.
while True:
    tr_num, tr_den = 0, 0
    for _ in range(100):
        classifier = classifier.train()
        input_x, input_y, input_t, _ = generator.sample_from_buffer(128)
        input_x, input_y, input_t  = input_x.to(args.device), input_y.to(args.device), input_t.to(args.device)

        opt_class.zero_grad()
        logits = classifier(input_x)

        if args.multiple_heads:
            mask = torch.zeros_like(logits)
            mask.scatter_(1, loader_cl.dataset.task_ids[input_t], 1)
            logits  = logits.masked_fill(mask == 0, -1e9)

        loss_class = F.cross_entropy(logits, input_y)
        loss_class.backward()
        opt_class.step()

        tr_num += logits.max(dim=-1)[1].eq(input_y).sum().item()
        tr_den += logits.size(0)

    tr_acc = tr_num / tr_den
    print('training acc : {:.4f}'.format(tr_acc))
    valid_acc = eval_cls('valid')

    if valid_acc > last_valid_acc:
        save_path = join(args.log_dir, 'cls.pth')
        print('saving cls to %s' % save_path)
        torch.save(classifier.state_dict(), save_path)
        last_valid_acc = valid_acc
    elif tr_acc > 0.99:
        break

# log the last classifier accuracy
writer.add_scalar('valid_classifier_acc', last_valid_acc, 0)

# make histogram with all the values
hist = make_histogram(RESULTS[-1, 0, -1], 'Valid Accuracy')
writer.add_image('Valid. Set Acc', hist, 0)

import time
time.sleep(10)
