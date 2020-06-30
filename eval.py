import wandb
import torch
import numpy as np
from torch.nn import functional as F
from torchvision.utils import save_image, make_grid

from utils.data   import *
from utils.buffer import *
from utils.utils  import dotdict, set_seed, get_chamfer
from utils.args   import get_args

from common.modular import QStack
from common.model   import ResNet18


@torch.no_grad()
def eval_drift(aqm, loader, args, log=True):

    all_loaders = list(loader)

    imgs    = {block.id: None for block in aqm.blocks}
    drifts  = {block.id: []   for block in aqm.blocks}
    drifts0 = {block.id: []   for block in aqm.blocks}

    if sum(block.frozen_qt for block in aqm.blocks) == 0: return

    for x, add_info in aqm.sample_everything():
        block_id = add_info['bid'][0]
        target = []

        if block_id == 0: continue

        target = []
        for _idx, _task in zip(add_info['bidx'], add_info['t']):
            loader = all_loaders[_task]
            if 'cifar' in args.dataset:
                target += [loader.dataset.rescale(loader.dataset.x[_idx])]
            else:
                target += [loader.dataset.__getitem__(_idx.item())[0]]

        target = torch.stack(target).to(x.device)

        img = torch.stack((x, target)).transpose(1,0).reshape(-1, *x.shape[1:])
        imgs[int(block_id)] = (img * .5 + .5).cpu()
        drifts[int(block_id)] += [F.mse_loss(x, target)]

        if (add_info['t'] == 0).any():
            drifts0[int(block_id)] += [
                    F.mse_loss(x[add_info['t'] == 0], target[add_info['t'] == 0])
                ]

    for key in drifts.keys():
        if len(drifts0[key]) > 0:
            mean0 = sum(drifts0[key]) / len(drifts0[key])
            wandb.log({'drift0_%d' % key: mean0})

        if len(drifts[key]) > 0:
            mean  = sum(drifts[key]) / len(drifts[key])
            img   = imgs[key]
            wandb.log({'drift_%d' % key: mean})
            wandb.log({'img_drift_%d' % key:
                [wandb.Image(make_grid(img), caption='drift %d' % key)]})

            print('{} \t{:.4f}'.format(key, mean))


@torch.no_grad()
def eval_gen(name, aqm, loader, args, log=True, max_task=-1, epoch=-1):
    """ evaluate performance on held-out data """

    print('eval test')
    with torch.no_grad():

        for task_t, te_loader in enumerate(loader):
            # only eval on seen tasks
            if task_t > max_task >= 0:
                break

            for data, target, _ in te_loader:
                data, target = data.to(args.device), target.to(args.device)

                all_recons, block_outs = aqm(data)

                for block_id in block_outs.keys():
                    block = aqm.all_blocks[block_id]
                    block_out = block_outs[block_id]

                    if log:
                        block.log('avg_l2_test', F.mse_loss(data, block_out['x_final']))

        if log:
            all = torch.cat((data.unsqueeze(0), all_recons))[:, :20]
            all = all.transpose(1,0).reshape(-1, *all.shape[2:])
            all = make_grid(all, nrow=all_recons.size(0) + 1)

            wandb.log({'test_recon_%d' % (epoch + max_task):
                [wandb.Image(make_grid(all), caption='test recon')]})

            aqm.log_to_server(wandb)


@torch.no_grad()
def eval_gen_lidar(name, generator, loader, args, max_task=-1, epoch=-1):
    """ evaluate performance on held-out data """

    chamfer = get_chamfer()

    print('eval test')
    with torch.no_grad():

        for task_t, te_loader in enumerate(loader):
            # only eval on seen tasks
            if task_t > max_task >= 0:
                break

            for data_raw, target, _ in te_loader:
                data_raw, target = data_raw.to(args.device), target.to(args.device)

                # normalize point cloud
                max_ = data_raw.reshape(data_raw.size(0), -1).abs().max(dim=1)[0].view(-1, 1, 1, 1)
                data = data_raw / max_

                all_recons, block_outs = generator(data)

                for block_id in block_outs.keys():
                    block = generator.all_blocks[block_id]
                    block_out = block_outs[block_id]

                    block.log('avg_l2_test', F.mse_loss(data, block_out['x_final']))

                    dist_a, dist_b  = chamfer(data_raw, block_out['x_final'] * max_)
                    snnrmse = (.5 * dist_a.mean(-1) + .5 * dist_b.mean(-1)).sqrt().mean()

                    block.log('snnrmse', snnrmse)

        all = torch.cat((data.unsqueeze(0), all_recons))[:, :20]

        all = all.transpose(1,0).reshape(-1, *all.shape[2:])
        all = make_grid(all, nrow=all_recons.size(0) + 1)

        wandb.log({'test_recon_%d' % (epoch + max_task):
            [wandb.Image(make_grid(all), caption='test recon')]})

        generator.log_to_server(wandb)


@torch.no_grad()
def eval_cls(classifier, loader, args, log=True, name='eval', max_task=-1):

    classifier.eval()

    accs  = np.zeros((max_task+1) if max_task > -1 else len(loader))
    denos = np.zeros_like(accs)

    for task_t, loader_t in enumerate(loader):
        # only eval on seen tasks
        if task_t > max_task >= 0:
            break

        for data, target, _ in loader_t:
            data, target = data.to(args.device), target.to(args.device)

            logits = classifier(data)

            if args.multiple_heads:
                logits = logits.masked_fill(loader_t.dataset.mask==0, -1e9)

            pred = logits.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum()
            deno = data.size(0)

            accs[task_t]  += correct
            denos[task_t] += deno

    accs = accs / denos

    if log:
        wandb.log({'%s_acc' % name : accs.mean()})

    return accs


