import os
import sys
import pdb
import torch
from torch import nn
from copy import deepcopy
from torch.nn import functional as F

sys.path += ['../']

from utils.utils     import *
from utils.buffer    import *
from common.quantize import Quantize
from common.model    import Encoder, Decoder

from torchvision.utils import save_image
from PIL import Image


# Quantization Building Block
# ------------------------------------------------------

class QLayer(nn.Module):
    def __init__(self, id, in_channel, channel, argmin_shp, data_shp, n_classes, n_res_blocks=1, downsample=2,
            n_embeds=128, n_codebooks=1, lr=1e-3, decay=0.6, dummy=False, opt='greedy',  **kwargs):

        super().__init__()

        self.id        = id
        self.avg_comp  = 0.
        self.avg_l2    = 0.
        self.logger    = RALog()
        self.log       = self.logger.log
        self.opt       = None

        if downsample > 1 and not dummy:
            # build networks
            self.encoder = Encoder(in_channel, channel, downsample, n_res_blocks)
            self.decoder = Decoder(channel, in_channel, downsample, n_res_blocks)

            if opt == 'greedy':
                self.opt = torch.optim.Adam(self.parameters(), lr=lr)

                if kwargs.get('lidar_mode', False):
                    self.opt = torch.optim.SGD(self.encoder.parameters(), lr=lr)
        else:
            self.encoder = self.decoder = self.ema_decoder = lambda x : x

        if dummy:
            self.buffer = Buffer(data_shp, n_classes, dtype=torch.FloatTensor)
            self.mem_per_sample = np.prod(data_shp)
            self.comp_rate = 1
            return

        # build quantization blocks
        D, K, N = channel, n_embeds, n_codebooks
        self.quantize = Quantize(D // N, K, N, decay=decay)

        # whether or not embedding matrix is frozen
        self.K           = K
        self.frozen_qt   = False
        self.downsample  = downsample

        argmin_shp = [n_codebooks] + argmin_shp
        self.buffer = Buffer(argmin_shp, n_classes, max_idx=n_embeds)
        self.mem_per_sample = self.buffer.mem_per_sample

        self.comp_rate   = np.prod(data_shp) / np.prod(argmin_shp) * np.log2(256) / np.log2(K)
        print('block ({})\t comp rate : {:.4f}'.format(self.id, self.comp_rate))

        self.size_in_floats = sum(np.prod(p.size()) for p in self.parameters())

        assert -.01 < (self.buffer.mem_per_sample - np.prod(data_shp) / self.comp_rate) < .01


    @property
    def n_samples(self):
        return self.buffer.n_samples


    @property
    def n_memory(self):
        return self.buffer.n_memory


    def track(self):
        self.logger.log('comp_rate', self.comp_rate)
        self.logger.log('n_samples', self.n_samples)
        self.logger.log('n_memory',  self.n_memory)
        self.logger.log('avg_comp',  self.avg_comp)
        self.logger.log('avg_l2',    self.avg_l2)
        self.logger.log('frozen',    self.frozen_qt)


    def update_ema_decoder(self):
        if not self.frozen_qt:
            return

        decay = .99
        try:
            for ema_param, param in zip(self.ema_decoder.parameters(), self.decoder.parameters()):
                ema_param.data.copy_(decay * ema_param.data + (1. - decay) * param.data)
        except:
            pass


    def init_ema(self):
        self.ema_decoder = deepcopy(self.decoder)

        self.size_in_floats += sum(np.prod(p.size()) for p in self.ema_decoder.parameters())


    def sample_everything(self, **kwargs):
        for argmin, add_info in self.buffer.sample_everything():
            z_q = self.quantize.idx_2_hid(argmin) if hasattr(self, 'quantize') else argmin

            block_id = torch.LongTensor(z_q.size(0)).fill_(self.id).to(z_q.device)
            add_info['bid'] = block_id

            yield z_q, add_info


    def sample(self, **kwargs):
        argmin, add_info = self.buffer.sample(**kwargs)
        z_q = self.quantize.idx_2_hid(argmin) if hasattr(self, 'quantize') else argmin

        block_id = torch.LongTensor(z_q.size(0)).fill_(self.id).to(z_q.device)
        add_info['bid'] = block_id

        return z_q, add_info


    def up(self, x):
        """ Encoding process """

        # downsample
        z_e   = self.encoder(x)

        # Used to be 75 --> now 90 --> now 95
        if self.avg_comp > .90 and not self.frozen_qt:
            self.quantize.decay = 1.
            n_embeds = self.quantize.trim()
            self.buffer.adjust_n_embeds(n_embeds)
            self.frozen_qt = True
            self.init_ema()

            self.comp_rate *= np.log2(self.K) / np.log2(n_embeds)
            print('fixing Block %d' % self.id)
            print('new comp rate : {:.4f}'.format(self.comp_rate))

        # quantize
        z_q, diff, argmin, ppl = self.quantize(z_e)

        output = {'x': x, 'z_e': z_e, 'z_q': z_q, 'ppl': ppl, 'diff': diff, 'argmin': argmin}

        self.log('ppl', ppl)
        self.log('argmin_unique', argmin.unique().size(0))

        return z_q, output


    def down(self, z):
        """ Decoding Process """

        return  self.decoder(z)


# Quantization Network (stack QLayers)
# ------------------------------------------------------

class QStack(nn.Module):
    def __init__(self, mem_args, block_args, data_args, opt_args, **kwargs):
        super().__init__()

        self.logger = RALog()
        self.log    = self.logger.log

        assert opt_args['opt'] in ['greedy', 'global']

        """ assumes args is a nested dictionary, one for every block """
        blocks = []

        # start at once, so 0 == uncompressed
        i = 1
        for block in sorted(block_args):
            blocks += [QLayer(i, **dict(block_args[block], **data_args, **opt_args))]
            i += 1

        # create a dummy block holding uncompressed data
        self.dummy = QLayer(0, 0, 0, None, dummy=True, **data_args)

        # only trainable blocks
        self.blocks = nn.ModuleList(blocks)

        # all (even placeholder) blocks
        self.all_blocks = [self.dummy] + [block for block in self.blocks]

        # optimization setup
        self.opt         = opt_args['opt']
        self.input       = opt_args['input']
        self.commit_coef = opt_args['commit_coef']
        self.recon_loss  = F.l1_loss if opt_args.get('recon_loss', '') == 'l1' else F.mse_loss

        if self.opt == 'global':
            self.global_opt = torch.optim.Adam(self.parameters(), lr=opt_args['global_lr'])

        # memory setup
        n_classes = data_args['n_classes']
        data_shp  = data_args['data_shp']

        self.recon_th = mem_args['recon_th']
        self.n_blocks = len(self.blocks)

        self.data_size  = np.prod(data_shp)
        self.mem_size_  = mem_args['mem_size'] * self.data_size * n_classes  # in floats

        self.register_buffer('mem_per_block', torch.Tensor([block.mem_per_sample for block in self.all_blocks]))


    @property
    def n_samples(self):
        return sum(block.n_samples for block in self.all_blocks)


    @property
    def mem_used(self):
        return sum(block.n_memory for block in self.all_blocks)


    @property
    def mem_size(self):
        mem_size =  self.mem_size_ - sum(block.size_in_floats for block in self.blocks)
        assert mem_size > 0

        return mem_size


    def track(self):
        self.log('n_samples', self.n_samples)
        self.log('mem_used',  self.mem_used / self.mem_size)

        for block in self.blocks: block.track()


    def log_to_server(self, wandb):
        wandb.log(self.logger.avg_dict())
        for block in self.blocks: wandb.log(block.logger.avg_dict(prefix=str(block.id)))

        self.logger.reset()
        for block in self.blocks: block.logger.reset()


    def update_ema_decoder(self):
        """ update the `old decoders` copy for every block """

        for block in self.blocks:
            block.update_ema_decoder()


    def up(self, x):
        """ Encoding process """

        block_outs = {}

        for i, block in enumerate(self.blocks):

            if i > 0:
                x = last_same_size_z
                if self.opt == 'greedy':
                    x = x.detach()

            x, block_out = block.up(x)
            block_outs[block.id] = block_out

            if i == 0 or block.downsample > 1:
               last_same_size_z = block_out[self.input]

        return x, block_outs


    def down(self, x, block_outs, decode_all=True):
        """ Decoding Process """

        n_og_samples = x.size(0)

        for i, block in enumerate(reversed(self.blocks)):
            block_out = block_outs[block.id]

            input = x
            if i > 0:
                if self.opt == 'greedy':
                    input = x.detach()

                if decode_all:
                    input = torch.cat((block_out['z_q'], input))

            x = block.down(input)
            block_out['x_hat'] = x[:n_og_samples]

        # (N, B, C, H, W) block_0, block_1, ...
        x = x.view(self.n_blocks, n_og_samples, *x.shape[1:])

        # log the final output
        for i, block_id in enumerate(sorted(block_outs.keys())):
            block_outs[block_id]['x_final'] = x[i]

        return x, block_outs


    def forward(self, x_inc, x_re=None):

        if x_re is None:
            x = x_inc
        else:
            x = torch.cat((x_inc, x_re))

        x, block_outs = self.up(x)
        x, block_outs = self.down(x, block_outs)

        if x_re is not None:
            # split the tensors between incoming and rehearsal
            lens       = [x_inc.size(0), x_re.size(0)]
            block_outs = dict_split(block_outs, suffix=['_inc', '_re'], lens=lens)

        return x, block_outs


    def optimize(self, block_outs):
        """ Loss calculation """

        total_loss = 0.

        if self.opt == 'global':
            self.global_opt.zero_grad()

        for block in reversed(self.blocks):
            block_out = block_outs[block.id]

            if self.opt == 'greedy' and block.opt is not None:
                block.opt.zero_grad()

            if block.downsample > 1:

                recon = self.recon_loss(block_out['x_hat'], block_out['x'])
                diff  = block_out['diff'].mean()

                block.log('recon', recon)
                block.log('diff',  diff)

                # rehearse
                if 'x_re' in block_out:
                    recon_re = self.recon_loss(block_out['x_hat_re'], block_out['x_re'])
                    diff_re  = block_out['diff_re'].mean()

                    block.log('recon_re', recon_re)
                    block.log('diff_re',  diff_re)

                    recon += recon_re
                    diff  += diff_re

                loss = recon + self.commit_coef * diff

                if self.opt == 'global':
                    total_loss += loss
                else:
                    loss.backward()
                    block.opt.step()

        if self.opt == 'global':
            total_loss.backward()
            self.global_opt.step()


    @torch.no_grad()
    def add_to_buffer(self, x, add_info, block_outs, sample_x=None, sample_add_info=None):

        # (B, )    -1 : not in buffer, 0 : uncompressed, 1 : 1st compression ...
        B = x.size(0)

        # only adding incoming samples
        sample_buffer_id  = torch.zeros(B).to(x.device).fill_(-1).long()
        sample_buffer_idx = sample_buffer_id.clone()
        already_comp      = torch.BoolTensor(B).fill_(False).to(x.device)

        if sample_add_info is not None:
            B_re = sample_x.size(0)

            # assuming all samples are not in buffer
            x                 = torch.cat((x, sample_x))
            sample_buffer_id  = torch.cat((sample_buffer_id,  sample_add_info['bid']))
            sample_buffer_idx = torch.cat((sample_buffer_idx, sample_add_info['idx']))
            already_comp      = torch.cat((already_comp, sample_add_info['bid'] > 0))
            add_info          = dict_cat((add_info, sample_add_info), discard=['bid', 'idx'])

        moved = torch.zeros_like(already_comp)

        # further_compress = (self.mem_used / self.mem_size) > .95
        # if not further_compress:
        #     already_comp[sample_buffer_id == 0] = 1

        for block in reversed(self.all_blocks):
            if block.id == 0:
                valid_comp  = (sample_buffer_id == -1)
                to_be_added = tba = x
            else:
                out = block_outs[block.id]
                tba = out['argmin']

                full_mse = F.mse_loss(x, out['x_final'], reduction='none')
                full_mse = full_mse.view(x.size(0), -1).mean(-1)

                # update block comp rate
                valid_comp = (full_mse < self.recon_th)
                block.avg_comp = 0.01 * valid_comp.float().mean() + 0.99 * block.avg_comp
                block.avg_l2   = 0.01 * full_mse.mean() + 0.99 * block.avg_l2

                if not block.frozen_qt:
                    continue

            adding = valid_comp & ~already_comp & ~moved
            block.buffer.add(tba, add_info, idx=adding)

            removing = (sample_buffer_id == block.id) & moved
            block.buffer.free(idx=sample_buffer_idx[removing])

            moved = moved | adding


    def _fetch_y_counts(self, exclude_task=None):
        out = []
        for block in self.all_blocks:
            buffer = block.buffer
            if exclude_task is not None:
                out += [buffer.y[buffer.t != exclude_task].sum(0)]
            else:
                out += [buffer.y.sum(0)]

        return torch.stack(out)


    def _balanced_sample(self, valid_classes, num_samples):
        n_classes = valid_classes.max() + 1
        probs = torch.FloatTensor(int(n_classes)).fill_(0).to(valid_classes.device)
        probs[valid_classes] = 1 / valid_classes.unique().size(0)

        sample = torch.multinomial(probs, num_samples=num_samples, replacement=True)

        return sample.bincount(minlength=n_classes)


    @torch.no_grad()
    def balance_memory(self):

        mem_excess = self.mem_used - self.mem_size

        while mem_excess > 0:
            # fetch block y dists
            y_counts = self._fetch_y_counts() # (n_blocks + 1, n_classes)
            class_counts = y_counts.sum(0)
            buff_counts  = y_counts.sum(1)

            for block in self.all_blocks:
                if mem_excess <= 0: break

                block_removal = int(np.ceil(mem_excess / block.mem_per_sample))
                class_removed, mem_freed = block.buffer.try_and_remove(block_removal, class_counts)

                mem_excess   -= mem_freed
                class_counts -= class_removed

            mem_excess = self.mem_used - self.mem_size


    def add_reservoir(self, x, add_info, block_outs, sample_x=None, sample_add_info=None):
        self.add_to_buffer(x, add_info, block_outs, sample_x=sample_x, sample_add_info=sample_add_info)
        self.balance_memory()


    @torch.no_grad()
    def sample(self, n_samples, exclude_task=None):

        """ figure out from which blocks and labels to pull the samples """

        y_counts = self._fetch_y_counts(exclude_task=exclude_task) # (n_blocks, n_cls)
        y_count  = y_counts.sum(0)
        valid_ys = y_count.nonzero().squeeze(-1)
        n_cls    = valid_ys.size(0)

        # sample number of class instances
        assert y_count.sum() >= n_samples

        # edit: you would like to draw samples according to the empirical seen distribution
        # this way we would really mimic reservoir sampling.
        # problem arises early on in new tasks, we we might be short on samples for a task.
        per_cls_sample = self._balanced_sample(valid_ys, n_samples)

        # TODO: put this back
        # make sure we have enough from each class
        # assert (y_count[valid_ys] - per_cls_sample).min() >= 0

        # fetch samples prop. to amount in each buffer , shp: (n_blocks, <n_cls)
        inter_buffer_dist  = y_counts[:, valid_ys] / y_counts[:, valid_ys].sum(0).float()

        # TODO: make this better
        per_buf_cls_sample = [
                torch.multinomial(inter_buffer_dist[:, i],
                                  num_samples=per_cls_sample[i],
                                  replacement=True).bincount(minlength=self.n_blocks + 1)
                if per_cls_sample[i] > 0 else torch.zeros_like(inter_buffer_dist[:, 0]).long()
                for i in range(n_cls)
            ]

        per_buf_cls_sample = torch.stack(per_buf_cls_sample, 1)

        """ get the samples """

        input = None
        for block in reversed(self.all_blocks):
            block_samples = per_buf_cls_sample[block.id]

            if block_samples.sum() == 0 and input is None:
                continue

            z_q, block_sample = block.sample(y_samples=block_samples)

            # first time collecting samples
            if input is None:
                input    = z_q
                add_info = block_sample
            else:
                input    = torch.cat((z_q, input))
                add_info = dict_cat((block_sample, add_info))

            input = block.ema_decoder(input)

        return input, add_info


    def sample_everything(self):
        for block in reversed(self.all_blocks):
            for z_q, add_info in block.sample_everything():

                for block_ in self.all_blocks[::-1]:
                    if block_.id > block.id: continue
                    z_q = block_.ema_decoder(z_q)

                yield z_q, add_info


def sho(x):
    save_image(x * .5 + .5, 'tmp.png')
    Image.open('tmp.png').show()


if __name__ == '__main__':
    import sys
    import yaml

    config = yaml.load(open('../test_config.yaml'), Loader=yaml.FullLoader)
    aqm = QStack(**config)

    x = torch.FloatTensor(16, 3, 32, 32).normal_()

    out, block_outs = aqm(x)

    y = torch.zeros(16).long()
    in_idx = y
    t = 1
    s = 1
    add_info = (y, t, in_idx, s)

    aqm.blocks[0].frozen_qt = True
    aqm.blocks[1].frozen_qt = True

    aqm.add_to_buffer(x, add_info, block_outs)
    # def add_to_buffer(x, add_info, block_outs, sample_outs=None):
