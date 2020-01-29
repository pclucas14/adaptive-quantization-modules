import gym
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
sys.path.append('../../atari-representation-learning')

from utils.data   import *
from utils.buffer import *
from utils.utils  import *
from utils.args   import get_args

from common.modular import QStack
from common.model   import ResNet18

from atariari.benchmark.wrapper import AtariARIWrapper
from atariari.benchmark.episodes import get_episodes

np.set_printoptions(threshold=3)

args = get_args()
rescale_inv = (lambda x : x * 0.5 + 0.5)

# spawn writer
args.log_dir = join(args.run_dir, args.model_name)  + args.suffix
sample_dir   = join(args.log_dir, 'samples')
writer       = SummaryWriter(log_dir=join(args.log_dir, 'tf'))
writer.add_text('hyperparameters', str(args), 0)

assert tuple(args.data_size) == (3, 210, 160)

# fetch model and ship to GPU
generator  = QStack(args).to(args.device)

env_name = {'pitfall':"PitfallNoFrameskip-v4", 'pong':"PongNoFrameskip-v4", 'mspacman':"MsPacmanNoFrameskip-v4"}[args.rl_env]

tr_episodes, val_episodes,\
tr_labels, val_labels,\
test_episodes, test_labels = get_episodes(env_name=env_name,
                                     steps=50000,
                                     collect_mode="random_agent",
                                     color=True)

kwargs = {'all_levels_recon':True}
if args.optimization == 'global':
    # flow gradient through blocks when doing global optimization
    kwargs.update({'inter_level_gradient':True})

step = 0
for task, episode in enumerate(tr_episodes):

    print('task : {} / {}'.format(task, len(tr_episodes)))

    n_batches = int(np.ceil(len(episode) / args.batch_size))
    episode   = torch.stack(episode).to(args.device)

    # episode is [0, 255].
    episode = (episode.float() / 255.) - .5

    next_frame = episode[1:]
    episode    = episode[:-1]

    sample_amt = 0
    for batch in range(n_batches):
        if batch % 5 == 0 : print('  ', batch, ' / ', len(episode), end='\r')
        input_x = episode[batch * args.batch_size: (batch+1) * args.batch_size]
        next_x  = next_frame[batch * args.batch_size: (batch+1) * args.batch_size]

        if sample_amt > args.samples_per_task > 0 : break
        sample_amt += input_x.size(0)

        input_x  = input_x.to(args.device)
        input_y = idx_ = torch.ones(input_x.size(0)).to(args.device).long()

        for n_iter in range(args.n_iters):

            if task > 0 and args.rehearsal:
                re_x, re_y, re_t, re_idx, re_step = generator.sample_from_buffer(args.buffer_batch_size)
                data_x, data_y = torch.cat((input_x, re_x)), torch.cat((input_y, re_y))
            else:
                data_x, data_y = input_x, input_y

            kwargs['next_frame'] = next_x
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

        step += 1

    generator.log(task, writer=writer, mode='train', should_print=args.print_logs)


    '''
    # Test the model
    # -------------------------------------------------------------------------------
    # if task < 2 or (task % 7 == 0): eval_gen('valid', max_task=task, break_after=2)
    if task % 4 == 0 or task < 3 or task == 19:
        eval_drift(max_task=task)
        eval_gen('valid', max_task=task)
    '''

    if args.rehearsal:
        buffer_sample, by, bt, _, _ = generator.sample_from_buffer(64)
        # save_image(rescale_inv(buffer_sample), '../samples/buf_%s_%d.png' % (args.model_name, task), nrow=8)
        save_image(rescale_inv(buffer_sample), '../samples/buf_%s_%d.png' % ('atari', task), nrow=8)

