import gym
import sys
import numpy as np
import pickle as pkl
from os.path import join
from pydoc  import locate
from copy   import deepcopy
from collections import defaultdict
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torchvision.utils import save_image

np.seterr(all='raise')

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
save_data = True

if args.gen_weights:
    """ loading model and data from file """

    generator, args = load_model_from_file(args.gen_weights)

    # load data
    ((tr_episodes,   tr_labels,   tr_ids),
    (val_episodes,  val_labels,  val_ids),
    (test_episodes, test_labels, test_ids)) = \
        pkl.load(open(os.path.join(args.log_dir, 'all_data.pkl'), 'rb'))

    all_episodes = tr_episodes + val_episodes + test_episodes
    all_labels   = tr_labels   + val_labels   + test_labels
    all_ids      = tr_ids      + val_ids      + test_ids

    id_train = sum([len(x) for x in tr_episodes])
    id_val   = sum([len(x) for x in val_episodes])  + id_train
    id_test  = sum([len(x) for x in test_episodes]) + id_val
    print(id_train, id_val, id_test)

else:
    """ generating data and training model """

    # spawn writer
    args.model_name =  args.model_name + '_%s' % args.rl_env
    args.log_dir    = join(args.run_dir, args.model_name)  + args.suffix
    sample_dir      = join(args.log_dir, 'samples')
    writer          = SummaryWriter(log_dir=join(args.log_dir, 'tf'))
    writer.add_text('hyperparameters', str(args), 0)

    print_and_save_args(args, args.log_dir)
    print('logging into %s' % args.log_dir)
    maybe_create_dir(sample_dir)

    assert tuple(args.data_size) == (3, 210, 160)

    # fetch model and ship to GPU
    generator  = QStack(args).to(args.device)

    env_name = {'pitfall':"PitfallNoFrameskip-v4",
                'pong':"PongNoFrameskip-v4",
                'mspacman':"MsPacmanNoFrameskip-v4"}[args.rl_env]

    tr_episodes, val_episodes,\
    tr_labels, val_labels,\
    test_episodes, test_labels = get_episodes(env_name=env_name,
                                         steps=5000,
                                         collect_mode="random_agent",
                                         color=True)


    # assign unide id to every frame
    tr_ids, val_ids, test_ids = [], [], []

    def build_ids(id_, episodes):
        all_ids = []
        for epi in episodes:
            epi_ids = []
            for frame in epi:
                epi_ids += [id_]
                id_ += 1

            all_ids += [epi_ids]

        return id_, all_ids

    id_train, tr_ids   = build_ids(0,        tr_episodes)
    id_val,   val_ids  = build_ids(id_train, val_episodes)
    id_test,  test_ids = build_ids(id_val,   test_episodes)

    # wanna save the real data ?
    if save_data:
        all_of_it = ((tr_episodes,   tr_labels,   tr_ids),
                     (val_episodes,  val_labels,  val_ids),
                     (test_episodes, test_labels, test_ids))

        import pickle as pkl
        print('saving data')
        pkl.dump(all_of_it, open(os.path.join(args.log_dir, 'all_data.pkl'), 'wb'))
        print('done saving data')

    all_episodes = tr_episodes + val_episodes + test_episodes
    all_labels   = tr_labels   + val_labels   + test_labels
    all_ids      = tr_ids      + val_ids      + test_ids


    kwargs = {'all_levels_recon':True}
    if args.optimization == 'global':
        # flow gradient through blocks when doing global optimization
        kwargs.update({'inter_level_gradient':True})

    step = 0
    for task, (episode, episode_label)  in enumerate(zip(all_episodes, all_ids)):

        print('task : {} / {}'.format(task, len(tr_episodes)))

        n_batches     = int(np.ceil(len(episode) / args.batch_size))
        episode       = torch.stack(episode).to(args.device)
        episode_label = torch.from_numpy(np.array(episode_label)).long().to(args.device)

        # episode is [0, 255].
        episode = (episode.float() / 255.) - .5

        next_frame    = episode[1:]
        episode       = episode[:-1]
        episode_label = episode_label[:-1]

        sample_amt = 0
        for batch in range(n_batches):
            if batch % 5 == 0 : print('  ', batch, ' / ', len(episode), end='\r')
            input_x = episode[batch * args.batch_size: (batch+1) * args.batch_size]
            next_x  = next_frame[batch * args.batch_size: (batch+1) * args.batch_size]
            idx_    = episode_label[batch * args.batch_size: (batch+1) * args.batch_size]

            if input_x.size(0) == 0: continue

            if sample_amt > args.samples_per_task > 0 : break
            sample_amt += input_x.size(0)

            input_x  = input_x.to(args.device)
            input_y  = torch.ones(input_x.size(0)).to(args.device).long()

            for n_iter in range(1 if generator.blocks[0].frozen_qt else args.n_iters):

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


        if args.rehearsal:
            buffer_sample, by, bt, _, _ = generator.sample_from_buffer(64)
            # save_image(rescale_inv(buffer_sample), '../samples/buf_%s_%d.png' % (args.model_name, task), nrow=8)
            save_image(rescale_inv(buffer_sample), '../samples/buf_%s_%d.png' % (args.rl_env, task), nrow=8)


    # save model (and buffer)
    if not args.debug:
        save_path = join(args.log_dir, 'gen.pth')
        print('saving model to %s' % save_path)
        torch.save(generator.state_dict(), save_path)


""" If you want to sample from the model """
new_tr_episodes,   new_tr_labels   = [], []
new_val_episodes,  new_val_labels  = [], []
new_test_episodes, new_test_labels = [], []

if True:
    # then sample from it
    gen_iter = generator.sample_EVERYTHING()

    # flatten labels
    all_labels_flat = []
    all_xs_flat     = []

    compare = []

    for item_a, item_b in zip(all_episodes, all_labels):
        all_xs_flat     += item_a
        all_labels_flat += item_b

    for batch in gen_iter:

        x, y, _, task, og_id, block_id = batch

        x = x.cpu()

        for i in range(x.size(0)):
            sample       = x[i]
            sample_id    = og_id[i]
            sample_label = all_labels_flat[sample_id]

            og_sample    = all_xs_flat[sample_id]

            if sample_id < id_train:
                new_tr_episodes   += [sample]
                new_tr_labels     += [sample_label]
            elif id_train <= sample_id < id_val:
                new_val_episodes  += [sample]
                new_val_labels    += [sample_label]
            else:
                new_test_episodes += [sample]
                new_test_labels   += [sample_label]


import pdb; pdb.set_trace()
""" IMPORTANT NOTE """
# this is for train valid and test,

# tr_episodes is an array of arrays containing (3, 210, 160) tensors
# whereas new_tr_episodes is an array containines (3, 210, 160)
# you have that len(new_tr_episodes) == (ish) to sum([len(x) for x in tr_episodes])
# the `ish` is because we lose the last frame of every episode during encoding. see line 139



