import sys
import pdb
import yaml
import wandb
import numpy as np
from os.path import join
from pydoc  import locate
from copy   import deepcopy
from collections import defaultdict
from torch.nn import functional as F
from torchvision.utils import save_image, make_grid

sys.path += ['../']

from eval         import *
from utils.data   import *
from utils.buffer import *
from utils.utils  import dotdict, get_chamfer, load_model
from utils.args   import get_args

from common.modular import QStack
from common.model   import ResNet18

np.set_printoptions(threshold=3)

Mean = lambda x : sum(x) / len(x)
rescale_inv = (lambda x : x * 0.5 + 0.5)
chamfer = get_chamfer()

best_test = float('inf')

@torch.no_grad()
def img_compress(block_outs, data, ext='PNG'):
    from io import BytesIO
    from PIL import Image

    logs = {i:[] for i in block_outs.keys()}

    for block_id in block_outs.keys():
        argmins = block_outs[block_id]['argmin']

        for argmin in argmins:

            new_argmin = []
            for argmin_ in argmin:
                used = argmin.unique()

                cp = argmin_.clone()
                for i in range(used.size(0)):
                    argmin_[cp == used[i]] = i

                new_argmin += [argmin_]

            new_argmin = torch.stack(new_argmin)

            buffer = BytesIO()
            im = Image.fromarray(new_argmin.permute(1,2,0).cpu().data.numpy().squeeze().astype('uint8'))
            im.save(buffer, ext)
            n_bytes = buffer.getbuffer().nbytes

            logs[block_id] += [n_bytes]


    return {i:Mean(logs[i]) for i in logs.keys()}


@torch.no_grad()
def check_comp(block_outs, data_raw, loader, th=0.15):

    # normalize point cloud
    max_ = data_raw.reshape(data_raw.size(0), -1).abs().max(dim=1)[0].view(-1, 1, 1, 1)
    data = data_raw / max_

    comp = torch.cuda.LongTensor(data_raw.size(0)).fill_(0)
    err  = torch.cuda.FloatTensor(comp.size(0)).fill_(0)

    for block_id in block_outs.keys():
        block_out = block_outs[block_id]

        dist_a, dist_b  = chamfer(data_raw, block_out['x_final'] * max_)
        snnrmse = (.5 * dist_a.mean(-1) + .5 * dist_b.mean(-1)).sqrt()

        comp[snnrmse < th] = block_id
        err[snnrmse < th]  = snnrmse[snnrmse < th]

    return comp, err


# ------------------------------------------------------------------------------
# Train the model
# ------------------------------------------------------------------------------
def main(args):

    suffix = {'online': 'online', 'offline': 'prepro'}

    if type(args) == dict: args = dotdict(args)

    mode = args.mode
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    for run in range(args.n_runs):
        wandb.init(project='aqm_lidar_%s' % suffix[mode], name=args.name, config={'yaml':config, 'params':args}, reinit=True)

        # fetch data
        data = locate('utils.data.get_%s' % args.dataset)(args, mode=mode)

        # make dataloaders
        train_loader, valid_loader, test_loader  = [
                CLDataLoader(elem, args, train=t, shuffle=(mode=='offline')) for elem, t in \
                        zip(data, [True, False, False])
        ]

        # fetch model and ship to GPU
        generator  = QStack(**config)

        if mode == 'online':
            load_model(generator, config['gen_weights'])

        generator = generator.to(args.device)
        print(generator)

        print("number of generator  parameters:", \
                sum([np.prod(p.size()) for p in generator.parameters()]))

        step = 0

        counts = torch.cuda.LongTensor(len(generator.all_blocks)).fill_(0)

        for task, tr_loader in enumerate(train_loader):
            print('dataset has %d examples' % len(tr_loader))

            for epoch in range(args.num_epochs):
                generator.train()
                generator.log('epoch', epoch)
                sample_amt = 0

                for i, (input_x_raw, input_y, idx_) in enumerate(tr_loader):
                    if i % 5 == 0 : print('  ', i, ' / ', len(tr_loader), end='\r')

                    if sample_amt > args.samples_per_task > 0: break
                    sample_amt += input_x_raw.size(0)

                    # normalize point cloud
                    max_    = input_x_raw.reshape(input_x_raw.size(0), -1).abs().max(dim=1)[0].view(-1, 1, 1, 1)
                    input_x = input_x_raw / max_

                    input_x = input_x.to(args.device)
                    input_y = input_y.to(args.device)
                    idx_    = idx_.to(args.device)

                    for n_iter in range(args.n_iters):

                        sample_outs = re_x = None
                        if task > 0 and args.rehearsal:
                            re_x, sample_outs = \
                                    generator.sample(args.buffer_batch_size, exclude_task=task)

                        # TODO: check if we're sampling the right amount
                        out, block_outs = generator(input_x, x_re=re_x)
                        generator.optimize(block_outs)

                        if mode == 'online' and n_iter == 0:
                            bid, err = check_comp(block_outs, input_x_raw.to(args.device), tr_loader, th=0.15)
                            counts += bid.bincount(minlength=counts.size(0))

                        if (i + 1) % (500 // args.batch_size) == 0 and n_iter == 0:

                            if mode == 'online':
                                wandb.log({'bytes sent': (generator.mem_per_block * counts).sum().item()})

                                byte_count = img_compress(block_outs, input_x_raw)
                                wandb.log({'png bytes sent':
                                    (generator.mem_per_block[0] * counts[0]).sum().item() +
                                    sum(byte_count[i] * counts[i].item() for i in byte_count.keys()),
                                            'bytes sent':
                                    (generator.mem_per_block * counts).sum().item()
                                    })
                                wandb.log({'count_%d' % i : counts[i].item() for i in range(len(counts))})

                            generator.log_to_server(wandb)

                    # set the gen. weights used for sampling == current generator weights
                    generator.update_ema_decoder()
                    generator.track()

                    if args.rehearsal:
                        generator.add_reservoir(
                                input_x,
                                {'y': input_y, 't': task, 'bidx': idx_, 'step': step},
                                block_outs,
                                sample_x=re_x,
                                sample_add_info=sample_outs
                        )

                    step += 1

                # Test the model
                # ------------------------------------------------------------------
                generator.update_ema_decoder()

                # save some samples
                if not args.debug and epoch > 0:
                    name = args.config.split('/')[-1].split('.yaml')[0]
                    xx = torch.stack([input_x] + [ block_outs[k]['x_final'] for k in sorted(block_outs.keys())] )
                    np.save(open('lidars/%s' % name , 'wb'), xx.cpu().data.numpy(), allow_pickle=False)

                if not args.debug:
                    eval_gen_lidar('valid', generator, valid_loader, args, max_task=task, epoch=epoch)
                    if task % 2 == 0 or task < 2: eval_drift(generator, train_loader, args)

                if not args.debug and (epoch + 1) % 10 == 0:
                    # save model
                    try:    name = args.name
                    except: name = args.config

                    os.makedirs('/checkpoint/lucaspc/aqm/' + name, exist_ok=True)
                    save_path = os.path.join('/checkpoint/lucaspc/aqm/', name, 'gen_%d.pth' % epoch)
                    torch.save(generator.state_dict(), save_path)



if __name__ == '__main__':
    args = get_args()
    main(args)
