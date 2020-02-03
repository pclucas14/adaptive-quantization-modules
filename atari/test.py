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

assert tuple(args.data_size[-2:]) == (210, 160)

# fetch model and ship to GPU
generator  = QStack(args).to(args.device)

env_name = {'pitfall':"PitfallNoFrameskip-v4",
            'pong':"PongNoFrameskip-v4",
            'mspacman':"MsPacmanNoFrameskip-v4"}[args.rl_env]

tr_episodes, val_episodes,\
tr_labels, val_labels,\
test_episodes, test_labels = get_episodes(env_name=env_name,
                                     steps=args.mem_size,
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
    print('saving data')
    out_tr_ep  = torch.cat([torch.cat(x) for x in tr_episodes])
    out_tr_len = [len(x) for x in tr_episodes]
    torch.save(out_tr_ep, os.path.join(args.log_dir, 'real_tr_data.pth'))
    np.savetxt(os.path.join(args.log_dir, 'real_tr_lens.txt'), out_tr_len)
    pkl.dump(tr_labels, open(os.path.join(args.log_dir, 'real_tr_labels.pkl'), 'wb'))

    out_val_ep  = torch.cat([torch.cat(x) for x in val_episodes])
    out_val_len = [len(x) for x in val_episodes]
    torch.save(out_val_ep, os.path.join(args.log_dir, 'real_val_data.pth'))
    np.savetxt(os.path.join(args.log_dir, 'real_val_lens.txt'), out_val_len)
    pkl.dump(val_labels, open(os.path.join(args.log_dir, 'real_val_labels.pkl'), 'wb'))

    out_test_ep  = torch.cat([torch.cat(x) for x in test_episodes])
    out_test_len = [len(x) for x in test_episodes]
    torch.save(out_test_ep, os.path.join(args.log_dir, 'real_test_data.pth'))
    np.savetxt(os.path.join(args.log_dir, 'real_test_lens.txt'), out_test_len)
    pkl.dump(test_labels, open(os.path.join(args.log_dir, 'real_test_labels.pkl'), 'wb'))

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
    episode       = torch.stack(episode)# .to(args.device)
    episode_label = torch.from_numpy(np.array(episode_label)).long().to(args.device)

    # episode is [0, 255].
    episode = ((episode.float() / 255.) - .5)*2.0

    next_frame    = episode[1:]
    episode       = episode[:-1]
    episode_label = episode_label[:-1]

    sample_amt = 0
    for batch in range(n_batches):
        if batch % 5 == 0 : print('  ', batch, ' / ', len(episode), end='\r')
        input_x = episode[batch * args.batch_size: (batch+1) * args.batch_size].to(args.device)
        next_x  = next_frame[batch * args.batch_size: (batch+1) * args.batch_size].to(args.device)
        idx_    = episode_label[batch * args.batch_size: (batch+1) * args.batch_size].to(args.device)

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
new_tr_episodes,   new_tr_labels  , new_tr_ids   = [], [], []
new_val_episodes,  new_val_labels , new_val_ids  = [], [], []
new_test_episodes, new_test_labels, new_test_ids = [], [], []

with torch.no_grad():
    # then sample from it
    gen_iter = generator.sample_EVERYTHING()

    # flatten labels
    all_labels_flat = []
    all_xs_flat     = []

    compare = []

    for item_a, item_b in zip(all_episodes, all_labels):
        all_xs_flat     += item_a
        all_labels_flat += item_b

    count = 0

    all_ = []
    for batch in gen_iter:

        x, _, _, _, og_id, _ = batch
        x=x.cpu()
        all_ += [x]

        x = x.cpu()
        og_id = og_id.cpu()

        for i in range(x.size(0)):
            count += 1
            if i % 1000 == 0: print('%d / %d' % (count, x.size(0)))

            sample       = x[i]
            sample_id    = og_id[i]
            sample_label = all_labels_flat[sample_id]

            og_sample    = all_xs_flat[sample_id]

            if sample_id < id_train:
                new_tr_episodes   += [sample]
                new_tr_labels     += [sample_label]
                new_tr_ids        += [sample_id]
            elif id_train <= sample_id < id_val:
                new_val_episodes  += [sample]
                new_val_labels    += [sample_label]
                new_val_ids       += [sample_id]
            else:
                new_test_episodes += [sample]
                new_test_labels   += [sample_label]
                new_test_ids      += [sample_id]


    sort_fn = lambda x: sorted(x, key=lambda v: v[0])

tr    = sort_fn(zip(new_tr_ids, new_tr_labels, new_tr_episodes))
val   = sort_fn(zip(new_val_ids, new_val_labels, new_val_episodes))
test  = sort_fn(zip(new_test_ids, new_test_labels, new_test_episodes))

new_tr_ids, new_tr_labels, new_tr_episodes          = [x[0] for x in tr],   [x[1] for x in tr],   [x[2] for x in tr]
new_val_ids, new_val_labels, new_val_episodes       = [x[0] for x in val],  [x[1] for x in val],  [x[2] for x in val]
new_test_ids, new_test_labels, new_test_episodes    = [x[0] for x in test], [x[1] for x in test], [x[2] for x in test]

# split the same way as the test set:
def split(labels, eps, ref):
    out_ep, out_label = [], []
    for ref_ep in ref:

        # -1 because of hte next frame being removed. see line 139
        last_idx = len(ref_ep) - 1
        out_ep    += [eps[:last_idx]]
        out_label += [labels[:last_idx]]

        eps = eps[last_idx:]
        labels = labels[last_idx:]

    return out_ep, out_label

sqm_tr_ep,   sqm_tr_label   = split(new_tr_labels,   new_tr_episodes,   tr_episodes)
sqm_val_ep,  sqm_val_label  = split(new_val_labels,  new_val_episodes,  val_episodes)
sqm_test_ep, sqm_test_label = split(new_test_labels, new_test_episodes, test_episodes)

out_tr_ep  = (rescale_inv(torch.cat([torch.cat(x) for x in sqm_tr_ep])) * 255.).clamp(0,255).round().byte()
out_tr_len = [len(x) for x in sqm_tr_ep]
torch.save(out_tr_ep, os.path.join(args.log_dir, 'sqm_tr_data.pth'))
np.savetxt(os.path.join(args.log_dir, 'sqm_tr_lens.txt'), out_tr_len)
pkl.dump(sqm_tr_label, open(os.path.join(args.log_dir, 'sqm_tr_labels.pkl'), 'wb'))



out_val_ep  = (rescale_inv(torch.cat([torch.cat(x) for x in sqm_val_ep])) * 255.).clamp(0,255).round().byte()
out_val_len = [len(x) for x in sqm_val_ep]
torch.save(out_val_ep, os.path.join(args.log_dir, 'sqm_val_data.pth'))
np.savetxt(os.path.join(args.log_dir, 'sqm_val_lens.txt'), out_val_len)
pkl.dump(sqm_val_label, open(os.path.join(args.log_dir, 'sqm_val_labels.pkl'), 'wb'))

out_test_ep  = (rescale_inv(torch.cat([torch.cat(x) for x in sqm_test_ep])) * 255.).clamp(0,255).round().byte()
out_test_len = [len(x) for x in sqm_test_ep]
torch.save(out_test_ep, os.path.join(args.log_dir, 'sqm_test_data.pth'))
np.savetxt(os.path.join(args.log_dir, 'sqm_test_lens.txt'), out_test_len)
pkl.dump(sqm_test_label, open(os.path.join(args.log_dir, 'sqm_test_labels.pkl'), 'wb'))

