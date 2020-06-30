import os
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

from utils.data   import *
from utils.buffer import *
from utils.utils  import dotdict, set_seed
from utils.args   import get_args

from common.modular import QStack
from common.model   import ResNet18
from eval import *

np.set_printoptions(threshold=3)

Mean = lambda x : sum(x) / len(x)
rescale_inv = (lambda x : x * 0.5 + 0.5)

best_test = float('inf')

def sho(x):
    save_image(x * .5 + .5, 'tmp.png')
    Image.open('tmp.png').show()


# ------------------------------------------------------------------------------
# Train the model
# ------------------------------------------------------------------------------
def main(args):

    if type(args) == dict: args = dotdict(args)
    data = locate('utils.data.get_%s' % args.dataset)(args)
    RESULTS = np.zeros((args.n_runs, 2, args.n_tasks, args.n_tasks))
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    for run in range(args.n_runs):

        set_seed(run)

        name = 'debug' if args.debug else args.config.split('/')[-1]
        wandb.init(project='aqm_lite_cifar', name=name, config={'yaml':config, 'params':args}, reinit=True)

        # fetch data
        data = locate('utils.data.get_%s' % args.dataset)(args)

        # make dataloaders
        train_loader, valid_loader, test_loader  = [
                CLDataLoader(elem, args, train=t) for elem, t in \
                        zip(data, [True, False, False])
        ]

        # fetch model and ship to GPU

        generator  = QStack(**config).to(args.device)
        classifier = ResNet18(args.n_classes, 20, input_size=args.input_size)
        classifier = classifier.to(args.device)
        print(generator)

        # optimizers
        opt_class = torch.optim.SGD(classifier.parameters(), lr=args.cls_lr)

        print("number of generator  parameters:", \
                sum([np.prod(p.size()) for p in generator.parameters()]))
        print("number of classifier parameters:", \
                sum([np.prod(p.size()) for p in classifier.parameters()]))

        step = 0
        for task, tr_loader in enumerate(train_loader):

            for epoch in range(args.num_epochs):
                generator.train()
                classifier.train()
                sample_amt = 0

                # create logging containers
                train_log = defaultdict(list)

                for i, (input_x, input_y, idx_) in enumerate(tr_loader):
                    if i % 5 == 0 : print('  ', i, ' / ', len(tr_loader), end='\r')

                    if sample_amt > args.samples_per_task > 0: break
                    sample_amt += input_x.size(0)

                    input_x = input_x_og = input_x.to(args.device)
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

                        if n_iter < args.cls_n_iters:
                            opt_class.zero_grad()
                            logits = classifier(input_x)

                            if args.multiple_heads:
                                mask = tr_loader.dataset.mask
                                logits = logits.masked_fill(mask == 0, -1e9)

                            opt_class.zero_grad()
                            loss_class = F.cross_entropy(logits, input_y)
                            loss_class.backward()

                            if args.rehearsal and task > 0:
                                logits = classifier(re_x)

                                if args.multiple_heads:
                                    mask = torch.zeros_like(logits)
                                    task_ids = tr_loader.dataset.task_ids[sample_outs['t']]
                                    mask.scatter_(1, task_ids, 1)
                                    logits  = logits.masked_fill(mask == 0, -1e9)

                                loss_class = F.cross_entropy(logits, sample_outs['y'])
                                loss_class.backward()

                            opt_class.step()

                        if (i + 1) % 50 == 0:
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

            print(generator._fetch_y_counts()[:, :(task + 1) * args.n_classes_per_task])
            print(generator.mem_used / generator.data_size)

            val_acc  = eval_cls(classifier, valid_loader, args, name='valid',  max_task=task)
            test_acc = eval_cls(classifier, test_loader, args, name='test', max_task=task)

            RESULTS[run, 0, task, :task+1] = val_acc
            RESULTS[run, 1, task, :task+1] = test_acc

            print(RESULTS[run, 0, :task+1, :task+1])

            if task % 2 == 0 or task < 2:
                eval_gen('valid', generator, valid_loader, args, max_task=task, epoch=epoch)
                eval_drift(generator, train_loader, args)

        print(RESULTS[run, 0, -1].mean(), RESULTS[run, 1, -1].mean())

        # calculate forgetting:
        max_valid = RESULTS[run, 0].max(axis=0)
        fgt_valid = (max_valid - RESULTS[run, 0, -1])[:-1].mean()

        max_test = RESULTS[run, 1].max(axis=0)
        fgt_test = (max_test - RESULTS[run, 1, -1])[:-1].mean()

        wandb.log({'fgt_valid': fgt_valid,
                   'acc_valid': RESULTS[run, 0, -1].mean(),
                   'fgt_test':  fgt_test,
                   'acc_test':  RESULTS[run, 1, -1].mean()})

        if not args.debug:
            # save model
            os.makedirs('/checkpoint/lucaspc/aqm/' + args.name, exist_ok=True)
            save_path = os.path.join('/checkpoint/lucaspc/aqm/', args.name, 'gen.pth')
            torch.save(generator.state_dict(), save_path)


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)

