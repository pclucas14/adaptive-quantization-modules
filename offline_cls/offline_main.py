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
print('stop' in args.suffix)

Mean = lambda x : sum(x) / len(x)
rescale_inv = (lambda x : x * 0.5 + 0.5)

# spawn writer
args.log_dir = join(args.run_dir, args.model_name)  + args.suffix
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
        '''
        TODO:
            1) iterate over all the data in the buffer. Proceed level by level (no need for task by task for now)
            2) keep a drift measure for every level.
        '''

        generator.eval()
        gen_iter = generator.sample_EVERYTHING()

        all_loaders = list(train_loader)

        for batch in gen_iter:

            x_t, y_t, _, task_t, idx_t, block_id = batch

            if block_id == -1: continue

            target = []
            for _idx, _task in zip(idx_t, task_t):
                loader = all_loaders[_task]
                if 'cifar' in args.dataset:
                    target += [loader.dataset.rescale(loader.dataset.x[_idx])]
                else:
                    target += [loader.dataset.__getitem__(_idx.item())[0]]

            if 'kitti' in args.dataset:
                target = torch.from_numpy(np.stack(target))
            else:
                target = torch.stack(target).to(x_t.device)

            target = target.to(x_t.device)
            diff = (x_t - target).pow(2).mean(dim=(1,2,3))

            # keep track of the errors on the first task
            diff_t0 = diff[task_t == 0]
            if diff_t0.size(0) > 0:
                generator.blocks[0].log.log('drift_mse_task0', diff_t0.mean(), per_task=False)

            # diff = diff[diff != 0.].mean()
            # remove nan
            # if diff != diff: diff = torch.Tensor([0.])

            generator.blocks[0].log.log('drift_mse_%d' % block_id, F.mse_loss(x_t, target), per_task=False)
            generator.blocks[0].log.log('drift_mse_total', F.mse_loss(x_t, target), per_task=False)

        generator.log(task, writer=writer, mode='buffer', should_print=True)


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

        print(RESULTS[-1, res_idx, -1].mean(), RESULTS[-1, res_idx, -1])
        return RESULTS

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

                # TODO: use ema
                outs = generator.reconstruct_all_levels(data)
                outs = generator.reconstruct_all_levels(data, **{'ema_decoder':True})
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



# ------------------------------------------------------------------------------
# Train generator
# ------------------------------------------------------------------------------

for run in range(args.n_runs):

    set_seed(args.seed)

    kwargs = {'all_levels_recon':True}

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

        step = 0
        for task, tr_loader in enumerate(train_loader):
            if task > args.max_task > -1: break

            for epoch in range(1):
                generator.train()
                sample_amt = 0

                print('task : {} / {}'.format(task, len(train_loader)))
                for i, (input_x, input_y, idx_) in enumerate(tr_loader):
                    if i % 5 == 0 : print('  ', i, ' / ', len(tr_loader), end='\r')

                    if sample_amt > args.samples_per_task > 0 : break
                    sample_amt += input_x.size(0)

                    input_x, input_y, idx_ = input_x.to(args.device), input_y.to(args.device), idx_.to(args.device)

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
                            # generator.add_reservoir(input_x, input_y, task, idx_, step=step)
                            generator.add_to_buffer(input_x, input_y, task, idx_, step=step)

                    # set the gen. weights used for sampling == current generator weights
                    generator.update_ema_decoder()

                    if (i+1) % 200 == 0 or (i+1) == len(tr_loader):
                        generator.log(task, writer=writer, mode='train', should_print=args.print_logs)

                    step += 1

                # Test the model
                # -------------------------------------------------------------------------------
                # if task < 2 or (task % 7 == 0): eval_gen('valid', max_task=task, break_after=2)
                if task % 4 == 0 or task < 3 or task == 19:
                    eval_drift(max_task=task)
                    eval_gen('valid', max_task=task)

            if args.rehearsal:
                buffer_sample, by, bt, _, _ = generator.sample_from_buffer(64)
                save_image(rescale_inv(buffer_sample), '../samples/buf_%s_%d.png' % (args.model_name, task), nrow=8)

            # generator.cut_lr()

        # save model
        if not args.debug:
            save_path = join(args.log_dir, 'gen.pth')
            print('saving model to %s' % save_path)
            torch.save(generator.state_dict(), save_path)

generator.cuda()
generator.eval()

RESULTS = np.zeros((args.n_runs, 2, args.n_tasks, args.n_tasks))

# ------------------------------------------------------------------------------
# Train classifier
# ------------------------------------------------------------------------------

# optimizers
classifier = ResNet18(args.n_classes, 20, input_size=args.input_size).to(args.device)
print("number of classifier parameters:", sum([np.prod(p.size()) for p in classifier.parameters()]))
print('cls learning rate {:.4f}'.format(args.cls_lr))
opt_class = torch.optim.SGD(classifier.parameters(), lr=args.cls_lr, momentum=0.9)

last_valid_acc = 0.
epoch = 0
while not args.debug or epoch < 1:
    epoch += 1
    tr_num, tr_den = 0, 0
    for _ in range(1 if args.debug else 100):
        classifier = classifier.train()
        input_x, input_y, input_t, _, _= generator.sample_from_buffer(128)
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
    valid_acc = valid_acc[-1, 0, -1].mean()

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
Image.fromarray(hist.transpose(1,2,0)).save(os.path.join(args.log_dir, 'valid_acc.png'))
writer.add_image('Valid. Set Acc', hist, 0)

# test set eval
classifier.load_state_dict(torch.load(os.path.join(args.log_dir, 'cls.pth')))

logs = eval_cls('test')
np.savetxt(os.path.join(args.log_dir, 'test_acc.txt'), logs[-1, -1, -1])
writer.add_scalar('test_classifier_acc', logs[-1, -1, -1].mean(), 0)

np.save(os.path.join(args.log_dir, 'results'), RESULTS)

import time
time.sleep(10)
print('finished')
