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
args.model_name = 'EGU{}_NB{}_Comp{}_Coef{:.2f}_{}'.format(args.layers[0].embed_grad_update,
                                                          args.num_blocks,
                                                          ''.join(['{:.2f}^'.format(args.layers[i].comp_rate) for i in range(args.num_blocks)]),
                                                          args.layers[0].decay + args.layers[0].commitment_cost,
                                             np.random.randint(10000))

# spawn writer
log_dir    = join('runs_IM_128', 'test' if args.debug else args.model_name)
sample_dir = join(log_dir, 'samples')
writer     = SummaryWriter(log_dir=join(log_dir, 'tf'))
writer.add_text('hyperparameters', str(args), 0)

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
            RESULTS[run][0 if name == 'valid' else 1][max_task][task_t] = mean_acc

            print('task %d' % task_t)

        print(RESULTS[-1, 0, -1], RESULTS[-1, 0, -1].mean())
        return RESULTS[-1, 0, -1].mean()

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
                if i_ > break_after > 0: break

                outs = generator.reconstruct_all_levels(data)
                out = generator(data,    **kwargs)

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
    set_seed(run)

    # fetch data
    data = locate('data.get_%s' % args.dataset)(args)

    # make dataloaders
    train_loader, valid_loader, test_loader  = [CLDataLoader(elem, args, train=t) for elem, t in zip(data, [True, False, False])]

    kwargs = {'all_levels_recon':True}

    # fetch model and ship to GPU
    generator  = QStack(args).to(args.device)

    try:
        load_model(generator, args.gen_weights)
    except Exception as e:
        print(e, '\n\n')

        print("number of generator  parameters:", sum([np.prod(p.size()) for p in generator.parameters()]))

        for task, tr_loader in enumerate(train_loader):
            if task > 1 and args.debug: break
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
                            re_x, re_y, re_t, re_idx = generator.sample_from_buffer(input_x.size(0))
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
                if (task + 1) % 5 ==  0: eval_gen('valid', max_task=task, break_after=2)

            buffer_sample, by, bt, _ = generator.sample_from_buffer(64)
            save_image(rescale_inv(buffer_sample), 'samples/buf_%s_%d.png' % (args.model_name, task), nrow=8)

        print(RESULTS)

        # save model
        save_path = join(log_dir, 'gen.pth')
        print('saving model to %s' % save_path)
        torch.save(generator.state_dict(), save_path)

# for the masks
loader_cl = train_loader[0]
generator.cuda()
generator.eval()

# optimizers
classifier = ResNet18(args.n_classes, 20, input_size=args.input_size).to(args.device)
classifier.train()
print("number of classifier parameters:", sum([np.prod(p.size()) for p in classifier.parameters()]))
opt_class = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
task, epoch = 20, -1

# build conveniant valid set
val_x, val_y, val_t = [], [], []
for val_l in valid_loader:
    for x,y, t in val_l:
        val_x += [x]
        val_y += [y]
        val_t += [t]

val_x = torch.cat(val_x)
val_y = torch.cat(val_y)
val_t = torch.cat(val_t)




last_valid_acc = 0.
plt = 0
##bon
from PIL import Image
from torchvision.utils import make_grid
def topil(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    tensor = rescale_inv(tensor)
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im
import torchvision.transforms as transforms
normalize = transforms.Normalize(mean=[ 0 , 0, 0],
                                     std=[0.5671, 0.5696, 0.5461])

apply_aug = transforms.Compose([
    transforms.Resize(156),
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

apply_b = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
to_pil_image = transforms.ToPILImage()
from torch.nn import BatchNorm2d
bn = BatchNorm2d(3, momentum=0.9).cuda()
maxval = 0
valid_ds = torch.utils.data.TensorDataset(val_x, val_y, val_t)
valid_loader_off = torch.utils.data.DataLoader(valid_ds, batch_size=256, shuffle=True, drop_last=False, num_workers=0)
while True:
    if plt == 15:
        opt_class = torch.optim.SGD(classifier.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    elif plt == 25:
        opt_class = torch.optim.SGD(classifier.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4)
    elif plt == 35:
        opt_class = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif plt == 40:
        break
    tr_num, tr_den = 0, 0
    classifier.train()
    bn.train()
    for _ in range(500):
        input_x, input_y, input_t, _ = generator.sample_from_buffer(128)
        dat = input_x.data.cpu().numpy()
        input_x = torch.cat([apply_aug(topil(d)).unsqueeze(0) for d in input_x])
        input_x, input_y, input_t  = input_x.to(args.device), input_y.to(args.device), input_t.to(args.device)

        opt_class.zero_grad()
        logits = classifier(bn(input_x))

        if args.multiple_heads:
            mask = torch.zeros_like(logits)
            mask.scatter_(1, loader_cl.dataset.task_ids[input_t], 1)
            logits  = logits.masked_fill(mask == 0, -1e9)

        loss_class = F.cross_entropy(logits, input_y)
        loss_class.backward()
        opt_class.step()

        tr_num += logits.max(dim=-1)[1].eq(input_y).sum().item()
        tr_den += logits.size(0)
    plt+=1
    print(plt)
    print('training acc : {:.4f}'.format(tr_num / tr_den))


    val_num, val_den = 0, 0
    classifier.eval()
    bn.eval()
    with torch.no_grad():
        for input_x, input_y, input_t in valid_loader_off:
            input_x, input_y, input_t  = input_x.to(args.device), input_y.to(args.device), input_t.to(args.device)
            #[normalize(inp) for inp in input_x]
            opt_class.zero_grad()
            logits = classifier(bn(input_x))

            if args.multiple_heads:
                mask = torch.zeros_like(logits)
                mask.scatter_(1, loader_cl.dataset.task_ids[input_t], 1)
                logits  = logits.masked_fill(mask == 0, -1e9)

            loss_class = F.cross_entropy(logits, input_y)
            val_den += logits.size(0)
            val_num += logits.max(dim=-1)[1].eq(input_y).sum().item()

    print('valid acc : {:.4f}'.format(val_num / val_den))
    if val_num/float(val_den) > maxval:
        maxval=val_num/float(val_den)
        torch.save(classifier.state_dict(),args.gen_weights+'.t7')
eval_cls('test', break_after=-1)
'''
    valid_acc = eval_cls('valid', break_after=10)
    if valid_acc < last_valid_acc:
        break
    else:
        last_valid_acc = valid_acc

eval_cls('test',  break_after=-1)
np.save(join(log_dir, 'results'), RESULTS)
'''

'''
# Make train and val splits for the classifier from the generator's buffer.
generator = generator.cpu()
data_x, data_y, data_t = generator.sample_EVERYTHING()
import pdb; pdb.set_trace()
# shuffle the data
idx = torch.randperm(data_x.size(0))
data_x, data_y, data_t = data_x[idx], data_y[idx], data_t[idx]

split = min(500, int(data_x.size(0) * 0.9))

train_x, valid_x = data_x[:split], data_x[split:]
train_y, valid_y = data_y[:split], data_y[split:]
train_t, valid_t = data_t[:split], data_t[split:]

train_ds = torch.utils.data.TensorDataset(train_x, train_y, train_t)
valid_ds = torch.utils.data.TensorDataset(valid_x, valid_y, valid_t)

train_loader_off = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True,  num_workers=0)
valid_loader_off = torch.utils.data.DataLoader(valid_ds, batch_size=32, shuffle=True, drop_last=False, num_workers=0)


# Starting here we perform the offline eval

# optimizers
classifier = ResNet18(args.n_classes, 20, input_size=args.input_size).to(args.device)
opt_class = torch.optim.SGD(classifier.parameters(), lr=args.cls_lr, momentum=0.9)

last_valid_loss = 1e9
while True:
    for input_x, input_y, input_t in train_loader_off:
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

    valid_loss, deno = 0., 0
    with torch.no_grad():
        for input_x, input_y, input_t in valid_loader_off:
            input_x, input_y, input_t  = input_x.to(args.device), input_y.to(args.device), input_t.to(args.device)

            opt_class.zero_grad()
            logits = classifier(input_x)

            if args.multiple_heads:
                mask = torch.zeros_like(logits)
                mask.scatter_(1, loader_cl.dataset.task_ids[input_t], 1)
                logits  = logits.masked_fill(mask == 0, -1e9)

            loss_class = F.cross_entropy(logits, input_y)
            valid_loss += loss_class.item()
            deno += 1

    new_valid_loss = valid_loss / deno
    print('valid loss', new_valid_loss)

    if new_valid_loss < last_valid_loss:
        last_valid_loss = new_valid_loss
    else:
        break
'''

"""

# Save final results
acc_avg = RESULTS.mean(axis=0)
acc_std = RESULTS.std(axis=0)

final_valid = acc_avg[0][-1]
final_test  = acc_avg[1][-1]

forget = RESULTS.max(axis=2) - RESULTS[:, :, -1, :]

print('final valid:')
out = ''
for acc_, std_ in zip(acc_avg[0][-1], acc_std[0][-1]):
    out += '{:.2f} +- {:.2f}\t'.format(acc_, std_ * 2 / np.sqrt(args.n_runs))
print(out)

print('{:.2f} +- {:.2f}'.format(RESULTS[:, 0, -1, :].mean(), RESULTS[:, 0, -1, :].std() * 2 / np.sqrt(args.n_runs)))
print('{:.2f} +- {:.2f}'.format(forget[:, 0, :].mean(), RESULTS[:, 0, :].std() * 2 / np.sqrt(args.n_runs)))


print('final test:')
out = ''
for acc_, std_ in zip(acc_avg[0][-1], acc_std[0][-1]):
    out += '{:.2f} +- {:.2f}\t'.format(acc_, std_ * 2 / np.sqrt(args.n_runs))
print(out)

print('{:.2f} +- {:.2f}'.format(RESULTS[:, 1, -1, :].mean(), RESULTS[:, 1, -1, :].std() * 2 / np.sqrt(args.n_runs)))
print('{:.2f} +- {:.2f}'.format(forget[:, 1, :].mean(), RESULTS[:, 1, :].std() * 2 / np.sqrt(args.n_runs)))
"""
