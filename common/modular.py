import os
import torch
from torch import nn
from copy import deepcopy
from torch.nn import functional as F

from utils.utils     import RALog, make_histogram
from utils.buffer    import *
from common.quantize import Quantize, GumbelQuantize, AQuantize
from common.model    import Encoder, Decoder

from torchvision.utils import save_image
from PIL import Image

def sho(x):
    save_image(x * .5 + .5, 'tmp.png')
    Image.open('tmp.png').show()

# Quantization Building Block
# ------------------------------------------------------

class QLayer(nn.Module):
    def __init__(self, id, args):
        super().__init__()

        self.id      = id
        self.args    = args
        self.log     = RALog()

        # build networks
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        assert args.num_codebooks == len(args.quant_size), \
                'amt of codebooks must match with codebook stride'

        # build quantization blocks
        qt  = []
        for i, size in zip(range(args.num_codebooks), args.quant_size):
            if args.model == 'vqvae':
                qt += [Quantize(args.embed_dim // args.num_codebooks,
                            args.num_embeddings, # // size ** 2,
                            decay=args.decay, size=size,
                            embed_grad_update=args.embed_grad_update)]
            elif args.model == 'gumbel':
                qt += [GumbelQuantize(args.embed_dim // args.num_codebooks)]
            elif args.model == 'argmax':
                qt += [AQuantize(args.embed_dim // args.num_codebooks)]

        self.quantize = nn.ModuleList(qt)

        if args.optimization == 'blockwise':
            # build layer opt
            self.opt = torch.optim.Adam(self.parameters(), lr=args.learning_rate)

        # TODO: check if Discretized Mixture of Logistic would do better
        self.register_parameter('dec_log_stdv', \
                torch.nn.Parameter(torch.Tensor([0.])))

        if args.rehearsal:
            self.mem_per_sample = 0
            self.comp_rate   = args.comp_rate

            buffers = []
            for shp in self.args.argmin_shapes:
                buffers += [Buffer(shp, args.n_classes, dtype=torch.LongTensor, max_idx=args.num_embeddings)]
                self.mem_per_sample += buffers[-1].mem_per_sample

            self.buffer = nn.ModuleList(buffers)

            # same for now
            self.old_quantize = self.quantize
            self.old_decoder  = self.decoder

            assert -.01 < (self.mem_per_sample - np.prod(args.data_size) / self.comp_rate) < .01


    @property
    def n_samples(self):
        return self.buffer[0].n_samples


    @property
    def n_memory(self):
        return sum([buffer.n_memory for buffer in self.buffer])


    def update_unused_vectors(self):
        for qt in self.quantize:
            qt.update_unused()


    def update_old_decoder(self):
        """ updates the stale decoder weights with the new ones """

        self.old_decoder   = deepcopy(self.decoder)
        self.old_quantize  = deepcopy(self.quantize)

        for qt in self.old_quantize:
            qt.no_update = True


    def add_to_buffer(self, argmins, y, t, ds_idx, idx=None):
        """ adds indices to layer buffer """

        assert len(argmins) == len(self.buffer)

        if idx is None:
            idx = torch.ones_like(y)

        # TODO: MUST ENSURE THAT THE BUFFER SHUFFLING IS THE SAME ACROSS BUFFER
        # id is off by one. TODO: fix.
        swap_idx = torch.randperm(int(self.n_samples))[:(idx == 1).sum()]

        for argmin, buffer in zip(argmins, self.buffer):
            # TODO: check this
            # Update: the following is because in main,
            # data_x = cat((input_x, re_x)). and we want to fetch input_x
            #argmin = argmin[-y.size(0):]
            argmin = argmin[:y.size(0)]
            buffer.add(argmin[idx], y[idx], t, ds_idx[idx], swap_idx)


    def rem_from_buffer(self, n_samples=None, idx=None):
        """ only adding this header cuz all other methods have one """

        import pdb

        for buffer in self.buffer:
            buffer.free(n_samples, idx=idx)

        if idx is not None:
            assert n_samples is None
            n_samples = idx.size(0)

        assert self.n_samples == self.buffer[0].n_samples \
                == self.buffer[-1].bx.size(0), pdb.set_trace()


    def add_argmins(self, y, t, ds_idx, argmin_idx=None, last_n=None):
        """ adds new representations to the buffer.
            made to be used with `update_buffer_idx` """

        # note: these arguments should always be passed, at least for now
        assert argmin_idx is not None and last_n is not None

        swap_idx = torch.randperm(int(self.n_samples))[:y.size(0)]

        for argmin, buffer in zip(self.argmins, self.buffer):
            if last_n is not None:
                argmin = argmin[-last_n:]
            if argmin_idx is not None:
                argmin = argmin[argmin_idx]

            buffer.add(argmin, y, t, ds_idx, swap_idx)


    def update_buffer(self, buffer_idx, argmin_idx=None, last_n=None):
        """ update the latent indices stored in the buffer """

        # note: these arguments should always be passed, at least for now
        assert argmin_idx is not None and last_n is not None

        for argmin, buffer in zip(self.argmins, self.buffer):
            if last_n is not None:
                argmin = argmin[-last_n:]
            if argmin_idx is not None:
                argmin = argmin[argmin_idx]

            buffer.update(buffer_idx, argmin)


    def sample_from_buffer(self, n_samples, from_comp=False):
        """ only adding this header cuz all other methods have one """

        with torch.no_grad():
            n_samples = int(n_samples)

            idx = torch.randperm(self.n_samples)[:n_samples]
            self.sampled_indices = idx

            # store the last sampled indices for a potential update
            self.sampled_idx = idx

            #for i, (qt, buffer) in enumerate(zip(self.quantize, self.buffer)):
            for i, (qt, buffer) in enumerate(zip(self.old_quantize, self.buffer)):
                if i == 0:
                    out_x = [qt.idx_2_hid(buffer.bx[idx])]
                    out_y = buffer.by[idx]
                    out_t = buffer.bt[idx]
                    out_idx = buffer.bidx[idx]
                else:
                    out_x += [qt.idx_2_hid(buffer.bx[idx])]
                    assert (out_y - buffer.by[idx]).abs().sum() == 0
                    assert (out_t - buffer.bt[idx]).abs().sum() == 0
                    assert (out_idx - buffer.bidx[idx]).abs().sum() == 0

            return torch.cat(out_x, 1), out_y, out_t, out_idx


    def sample_EVERYTHING(self):
        """ only adding this header cuz all other methods have one """

        with torch.no_grad():

            idx = torch.randperm(self.n_samples)

            for i, (qt, buffer) in enumerate(zip(self.old_quantize, self.buffer)):
                if i == 0:
                    out_x = [qt.idx_2_hid(buffer.bx[idx])]
                    out_y = buffer.by[idx]
                    out_t = buffer.bt[idx]
                    out_idx = buffer.bidx[idx]
                else:
                    out_x += [qt.idx_2_hid(buffer.bx[idx])]
                    assert (out_y - buffer.by[idx]).abs().sum() == 0
                    assert (out_t - buffer.bt[idx]).abs().sum() == 0
                    assert (out_idx - buffer.bidx[idx]).abs().sum() == 0

            return torch.cat(out_x, 1), out_y, out_t, out_idx


    def idx_2_hid(self, indices):
        """ fetch latent representation from indices """

        assert len(indices) == len(self.quantize), pdb.set_trace()

        out = []
        for idx, qt in zip(indices, self.quantize):
            out += [qt.idx_2_hid(idx)]

        return torch.cat(out, dim=1)


    def up(self, x, **kwargs):
        """ Encoding process """

        # 1) encode
        z_e   = self.encoder(x)
        z_e_s = z_e.chunk(self.args.num_codebooks, 1)

        # 2) quantize
        z_q_s, argmins, ppls, diffs = [], [], [], []

        for z_e, quantize in zip(z_e_s, self.quantize):
            z_q, diff, argmin, ppl = quantize(z_e)

            z_q_s   += [z_q]
            diffs   += [diff]
            argmins += [argmin]
            ppls    += [ppl]

        z_q = torch.cat(z_q_s, dim=1)

        # save tensors required tensors for later
        self.z_e     = z_e
        self.z_q     = z_q
        self.ppls    = ppls
        self.diffs   = diffs
        self.argmins = argmins

        if not kwargs.get('no_log', False):
            # store as scalars for convenience
            for i in range(len(self.diffs)):
                self.log.log('ppl-B%d-C%d'  % (self.id, i), self.ppls[i],  per_task=False)
                self.log.log('diff-B%d-C%d' % (self.id, i), self.diffs[i], per_task=False)
        return z_q


    def down(self, x, **kwargs):
        """ Decoding Process """

        if kwargs.get('old_decoder', False):
            return self.old_decoder(x)

        self.output = self.decoder(x)

        return self.output


    def loss(self, target, **kwargs):
        """ Loss calculation """

        if kwargs.get('all_levels_recon', False):
            self.output = self.output[-target.size(0):]

        # TODO: should we weight these differently ?
        diffs = sum(self.diffs) / len(self.diffs)
        recon = F.mse_loss(self.output, target)

        # TODO: change that
        # recon = F.l1_loss(self.output, target)
        # recon = (self.output - target).abs().sum(dim=(1,2,3)).mean(dim=0)
        # import pdb; pdb.set_trace()

        self.recon = recon.item()
        if not kwargs.get('no_log', False):
            # during eval we actually perform a complete log, so no need for per-task here
            self.log.log('Distill_recon-B%d' % self.id, self.recon, per_task=False)
            pass

        return recon, diffs


# Quantization Network (stack QLayers)
# ------------------------------------------------------

class QStack(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.log_step = 0
        self.n_seen_so_far = 0
        self.rehearsal_level = -1

        self.register_buffer('recon_th', torch.Tensor(self.args.recon_th))

        """ assumes args is a nested dictionary, one for every block """
        blocks = []
        for layer_no in range(len(args.layers)):
            blocks += [QLayer(layer_no, args.layers[layer_no])]

        self.blocks = nn.ModuleList(blocks)

        if args.optimization == 'global':
            self.opt = torch.optim.Adam(self.parameters(), \
                    lr=args.global_learning_rate)

        if args.rehearsal:
            mem_size  = args.mem_size * np.prod(args.data_size)

            if not args.sunk_cost:
                mem_size -= sum(p.numel() for p in self.parameters())

            self.mem_size = mem_size    # total floats that can be stored across all blocks
            self.n_seen_so_far = 0      # number of samples seen so far

            self.all_stored_ = 0        # samples stored across all blocks
            self.mem_used_   = 0        # total float used across blocks

            self.data_size = np.prod(args.data_size)
            self.can_store_reg = mem_size // (self.data_size)

            # whether we need to recompute the buffer statistics
            self.up_to_date_mu = self.up_to_date_as  = True

            self.reg_buffer = Buffer(args.data_size, \
                    args.n_classes, dtype=torch.FloatTensor)

            comp_rate = [self.data_size / block.args.comp_rate for block in self.blocks]
            self.register_buffer('mem_per_block', torch.Tensor([self.data_size] + comp_rate))


    @property
    def all_stored(self):
        if self.up_to_date_as:
            return self.all_stored_

        total = self.reg_buffer.n_samples

        for block in self.blocks:
            total += block.n_samples

        self.all_stored_ = total
        self.up_to_date_as = True

        return total


    @property
    def mem_used(self):
        if self.up_to_date_mu:
            return self.mem_used_

        total = self.reg_buffer.n_memory

        for block in self.blocks:
            total += block.n_memory

        self.mem_used_ = total
        self.up_to_date_mu = True

        return total


    @property
    def reg_stored(self):
        return self.reg_buffer.n_samples


    def up_to_date(self, value):
        self.up_to_date_mu = self.up_to_date_as = value


    def update_old_decoder(self):
        """ update the `old decoders` copy for every block """

        for block in self.blocks:
            block.update_old_decoder()


    def buffer_update_idx(self, re_x, re_y, re_t, re_ds_idx):

        re_target = self.all_levels_recon[:, -re_x.size(0):]

        per_block_l2 = (re_x.unsqueeze(0) - re_target).pow(2)
        per_block_l2 = per_block_l2.mean(dim=(2,3,4))

        recon_th = self.recon_th.unsqueeze(1).expand_as(per_block_l2)
        block_id = (per_block_l2 < recon_th).sum(dim=0)

        # TODO: build block id when sampling
        pre_block_id = self.last_block_id.to(block_id.device)

        # take the most compressed rep (biggest block id)
        new_block_id = torch.stack((block_id, pre_block_id)).max(dim=0)[0]

        # first, delete points from real buffer which will be compressed
        delete_idx = self.sampled_indices[(pre_block_id == 0) * (new_block_id > 0)]
        self.reg_buffer.free(idx=delete_idx)

        # monitor what happens
        same_lvl_amt = (pre_block_id == new_block_id).sum()
        jump01_amt   = ((pre_block_id == 0) * (new_block_id == 1)).sum()
        jump12_amt   = ((pre_block_id == 1) * (new_block_id == 2)).sum()

        self.blocks[0].log.log('same_lvl', same_lvl_amt, per_task=False)
        self.blocks[0].log.log('jump01', jump01_amt, per_task=False)
        self.blocks[0].log.log('jump12', jump12_amt, per_task=False)

        for i, block in enumerate(self.blocks):

            if not self.args.no_idx_update:
                # 1) update stale representations
                update_mask = (pre_block_id == (i+1)) * (new_block_id == (i+1))
                update_idx  = self.sampled_indices[update_mask]
                block.update_buffer(update_idx, argmin_idx=update_mask, last_n=re_x.size(0))

            # 2) delete representations that will be further compressed
            delete_mask = (pre_block_id == (i+1)) * (new_block_id > (i+1))
            delete_idx  = self.sampled_indices[delete_mask]
            block.rem_from_buffer(idx=delete_idx)

            # 3) add new representations
            add_mask    = (pre_block_id < (i+1))  * (new_block_id == (i+1))
            block.add_argmins(re_y[add_mask], re_t[add_mask], re_ds_idx[add_mask], argmin_idx=add_mask, last_n=re_x.size(0))

        # will need to recompute statistics
        self.up_to_date(False)


    def sample_from_buffer(self, n_samples, exclude_task=None):
        """ something something something """

        # note: the block ids are off by one with block.id :/ i.e. block.id == 0 will be 1 here
        self.block_id = torch.zeros(n_samples).long()

        # sample proportional to the amounts of points per resolution
        probs = torch.Tensor([self.reg_stored] + [x.n_samples for x in self.blocks])
        probs = probs / probs.sum()
        samples_per_block = (probs * n_samples).floor()


        #TODO: this will throw an error if no samples are stored in full resolution
        missing = n_samples - samples_per_block.sum()

        if (missing + samples_per_block[0]) <= self.reg_buffer.n_samples:
            samples_per_block[0] += missing
        else:
            samples_per_block[samples_per_block.argmax()] += missing

        # keep track of this to update latent indices
        self.last_samples_per_block = samples_per_block
        self.last_block_id = torch.zeros(n_samples).long().fill_(len(self.blocks))

        # TODO: make this more efficient
        current_sum = 0
        for i in range(samples_per_block.size(0)):
            self.last_block_id[:current_sum] -= 1
            current_sum += int(samples_per_block[i].item())

        import pdb
        assert samples_per_block[0] <= self.reg_buffer.n_samples, pdb.set_trace()

        reg_x, reg_y, reg_t, reg_ds_idx = self.reg_buffer.sample(samples_per_block[0], exclude_task=exclude_task)

        # keep strack of the sampled indices
        self.sampled_indices = []

        if samples_per_block[0] == n_samples:
            self.sampled_indices = self.reg_buffer.sampled_indices
            return reg_x, reg_y, reg_t, reg_ds_idx

        # we reverse the blocks, so that all the decoding can be done in one pass
        r_blocks, r_spb = self.blocks[::-1], reversed(samples_per_block[1:])

        i = 0
        for (block_samples, block) in zip(r_spb, r_blocks):
            if block_samples == 0 and i == 0:
                continue

            xx, yy, tt, ds_idx  = block.sample_from_buffer(block_samples)

            self.sampled_indices = [block.sampled_indices] + self.sampled_indices

            if i == 0:
                out_x = xx
                out_y = yy
                out_t = tt
                out_idx = ds_idx
            else:
                out_x = torch.cat((xx, out_x))
                out_y = torch.cat((yy, out_y))
                out_t = torch.cat((tt, out_t))
                out_idx = torch.cat((ds_idx, out_idx))

            # use old weights when sampling
            out_x = block.old_decoder(out_x)

            i += 1

        # TODO: check if should be on CUDA already
        self.sampled_indices = torch.cat([self.reg_buffer.sampled_indices.cpu()] + \
                self.sampled_indices)

        return torch.cat((reg_x, out_x)), torch.cat((reg_y, out_y)), torch.cat((reg_t, out_t)), torch.cat((reg_ds_idx, out_idx))


    def sample_EVERYTHING(self):
        """ something something something """

        reg_x, reg_y, reg_t = self.reg_buffer.bx, self.reg_buffer.by, self.reg_buffer.bt

        # we reverse the blocks, so that all the decoding can be done in one pass
        r_blocks = self.blocks[::-1]

        i = 0
        for block in r_blocks:

            xx, yy, tt, ds_idx  = block.sample_EVERYTHING()
            if i == 0 and xx.size(0) == 0:
                continue

            if i == 0:
                out_x = xx
                out_y = yy
                out_t = tt
                out_idx = ds_idx
            else:
                out_x = torch.cat((xx, out_x))
                out_y = torch.cat((yy, out_y))
                out_t = torch.cat((tt, out_t))
                out_idx = torch.cat((ds_idx, out_idx))

            # use old weights when sampling
            outs = []
            for chunk in out_x.split(100):
                outs += [block.old_decoder(out_x)]

            out_x = torch.cat(outs)
            #out_x = block.old_decoder(out_x)

            i += 1

        return torch.cat((reg_x, out_x)).detach(), torch.cat((reg_y, out_y)).detach(), torch.cat((reg_t, out_t)).detach()



    def add_reservoir(self, x, y, t, ds_idx, **kwargs):
        """ Reservoir Sampling Buffer Addition """

        mem_free = self.mem_size - self.mem_used

        if x.size(0) > 0:
            # in reservoir sampling, samples should be added with
            # p(amt of samples that fit in mem / samples see so far)
            indices = torch.FloatTensor(x.size(0)).to(x.device).\
                    uniform_(0, self.n_seen_so_far).long()

            valid_indices = (indices < max(self.can_store_reg, self.all_stored)).long()

            # indices of samples to be added in mem
            # note that this process is independant of the compression rate
            # which should make things less biased.
            idx_new_data = valid_indices.nonzero().squeeze(-1)

            '''
            if self.blocks[0].buffer[0].n_samples > 100:
                import pdb; pdb.set_trace()
            '''

            # now that we know which samples will be added, we need to check
            # which rep / compression rate will be used.

            """ only using recon error for now """
            # keep in memory the last sampled indices
            # think about cleanest way to update the stored representations
            target = self.all_levels_recon[:, :x.size(0)]

            # now that we know which samples will be added to the buffer,
            # we need to find the most compressed representation that is good enough

            per_block_l2 = (x.unsqueeze(0) - target).pow(2)
            per_block_l2 = per_block_l2.mean(dim=(2,3,4))
            recon_th = self.recon_th.unsqueeze(1).expand_as(per_block_l2)
            block_id = (per_block_l2 < recon_th).sum(dim=0)

            # we calculate the amount of space that needs to be freed
            space_needed = F.one_hot(block_id[idx_new_data], len(self.blocks) + 1).float()
            space_needed = (space_needed * self.mem_per_block).sum()
            space_needed = (space_needed - mem_free).clamp_(min=0.)

            # for samples that will not be added, mark their block id as -1
            if idx_new_data.size(0) < x.size(0):
                ind = torch.ones(block_id.size(0))
                ind[idx_new_data] -= 1
                ind = ind.nonzero().squeeze()
                block_id[ind] = -1

            # we want the removal of samples in the buffer to be agnostic to the
            # compression rate. We determine how much to remove from every block
            # E[removed from b_i] = space_bi_takes / total_space * space_to_be_removed
            to_be_removed_weights = torch.Tensor([self.reg_buffer.n_memory] + \
                    [block.n_memory for block in self.blocks])

            if space_needed > 0:
                to_be_removed_weights = to_be_removed_weights / to_be_removed_weights.sum()
                tbr_per_block_mem = to_be_removed_weights * space_needed
                tbr_per_block_n_samples = (tbr_per_block_mem / self.mem_per_block.cpu()).ceil()
            else:
                tbr_per_block_n_samples = torch.zeros_like(to_be_removed_weights).long()

            # mark the buffer stats as needing update
            self.up_to_date(False)

            # finally, we iterate over the blocks and add / remove the required samples
            # 0th block (uncompressed)
            self.reg_buffer.free(tbr_per_block_n_samples[0])
            self.reg_buffer.add(x[block_id == 0], y[block_id == 0], t, ds_idx[block_id == 0], swap_idx=None)

            for i, block in enumerate(self.blocks):
                # free space
                block.rem_from_buffer(tbr_per_block_n_samples[i + 1])

                # add new points
                block.add_to_buffer(block.argmins, y, t, ds_idx, idx=(block_id == (i+1)))

            """ Making sure everything is behaving as expected """

            # update statistic
            self.n_seen_so_far += x.size(0)

            for block in self.blocks:
                block.log.log('buffer-samples-B%d' % block.id, block.n_samples, per_task=False)

            block.log.log('buffer_samples-reg', self.reg_stored, per_task=False)
            block.log.log('buffer-mem', self.mem_used, per_task=False)
            block.log.log('n_seen_so_far', self.n_seen_so_far, per_task=False)


    def log_buffer(self):
        """ Monitor label distribution in buffers """

        hist = torch.zeros(self.args.n_classes).long()

        hist += self.reg_buffer.y.sum(dim=0).cpu()

        self.blocks[0].log.log('buffer-y-count-REG', hist.clone(), per_task=False)

        # add the labels for all buffer levels
        for block in self.blocks:
            if block.n_samples > 0:
                block_hist = block.buffer[0].y.sum(dim=0).cpu()
                block.log.log('buffer-y-count-B%d' % block.id, block_hist, per_task=False)
                hist += block_hist

        hist = hist.float()
        hist = hist / hist.sum()

        block.log.log('buffer-y-dist', hist, per_task=False)


    def up(self, x, **kwargs):
        """ Encoding process """

        # you have two options here. You can either
        # 1) propagate gradient between levels
        # 2) treat every level as completely independant

        for block in self.blocks:
            if kwargs.get('inter_level_gradient', False):
                x = x               # option 1
            else:
                x = x.detach()      # option 2

            x = block.up(x, **kwargs)

        return x


    def down(self, x, **kwargs):
        """ Decoding Process """

        # you have two options here. You can either use as input
        # 1) the output of the stream from the bottom block
        # 2) the output of the quantizer from the same block
        # 3) do both at the same time in one fwd pass
        # UPDATE: actually always do 3, but whether gradient is like 1 or 2
        # can be done with a proper detach call

        for i, block in enumerate(reversed(self.blocks)):
            # TODO: figure out what behavior we want here
            ''''
            if kwargs.get('inter_level_stream', False):
                block.z_q = block.z_q.detach()
            else:
                x = x.detach()
            '''

            # removing deepest block as x == block.z_q for it
            if i == 0:
                input = x                           # option 2
            else:
                x = x.detach()
                input = torch.cat((x, block.z_q))   # option 1

            x = block.down(input, **kwargs)

        # original batch size
        n_og_samples = block.z_q.size(0)

        # if `all_levels_recon`, returns a tensor of shape
        # (bs * n_levels, C, H, W). can call `.view(n_levels, bs, ...)
        # to split correctly. Levels are ordered from deepest (top) to bot.
        # i.e. the last one will have the best reconstruction

        x = x.view(len(self.blocks), n_og_samples, *x.shape[1:])
        self.all_levels_recon = x

        # return only the "nicest" one
        return x[-1]


    def forward(self, x, **kwargs):
        x = self.up(x,   **kwargs)
        x = self.down(x, **kwargs)
        return x


    def optimize(self, target, **kwargs):
        """ Loss calculation """

        total_loss = 0.
        for i, block in enumerate(self.blocks):
            # TODO: check performance difference between using `z_q` vs `z_e`
            target_i = target if i == 0 else self.blocks[i-1].z_q

            # it's important to detach the target! (similar to RL / Q-learning)
            recon, diff = block.loss(target_i.detach(), **kwargs)

            loss = recon + block.args.commitment_cost * diff

            if self.args.optimization == 'global':
                total_loss += loss
            else:
                # optimize
                block.opt.zero_grad()

                # TODO: retain graph if inter_level_gradient is on
                loss.backward() #retain_graph=(i+1) != len(self.blocks))

                block.opt.step()

                #block.update_unused_vectors()

        if self.args.optimization == 'global':
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()


    def decode_indices(self, indices):
        """ fetch latent representation from indices """

        out_levels = []
        for i, block in enumerate(reversed(self.blocks)):
            # current block
            x = block.idx_2_hid(indices[i])
            x = block.decoder(x)
            for j, block_ in enumerate(self.blocks[::-1][i+1:]):
                x = block_.decoder(x)

            # store output
            out_levels += [x]

        return out_levels


    def fetch_indices(self):
        """ fetches the latest set of indices stored in block stack """

        # note: the returned array should be ordered for `decode_indices`
        # i.e. indices[0] == most nested / deepest block

        indices = []
        for block in reversed(self.blocks):
            indices += [block.argmins]

        return indices


    def reconstruct_all_levels(self, og, **kwargs):
        """ Expand all levels to input space to evaluate reconstruction """

        # encode
        self.up(og, **kwargs)

        out_levels = []
        for i, block in enumerate(reversed(self.blocks)):
            # current block
            x = block.decoder(block.z_q)
            for j, block_ in enumerate(self.blocks[::-1][i+1:]):
                x = block_.decoder(x)

            # store output
            out_levels += [x]

            # log reconstruction error
            #block.log.log('Full_recon-B%d' % block.id, F.l1_loss(x, og))
            block.log.log('Full_recon-B%d' % block.id, F.mse_loss(x, og))

        return out_levels


    def log(self, task, writer=None, should_print=False, mode='train'):
        """ Logs the results """

        # get buffer informations
        self.log_buffer()

        if writer is not None:
            for block in self.blocks:
                for name, value in block.log.storage.items():
                    prefix = mode + '/'
                    if block.log.per_task[name]:
                        suffix = '__task' + str(task)
                    else:
                        suffix = ''

                    if type(value) == np.ndarray:
                        # Tensorboard's histogram is awful. Let's make an image instead
                        tmp_path = os.path.join(self.args.log_dir, 'tmp.png')
                        hist = make_histogram(value, prefix + name + suffix, tmp_path=tmp_path)
                        writer.add_image(prefix + name + suffix, hist, self.log_step)
                    else:
                        writer.add_scalar(prefix + name + suffix, value, self.log_step)

            self.log_step += 1

        if should_print:
            print(prefix)
            for block in self.blocks:
                print(block.log.one_liner())
            print('\n')

        # reset logs
        for block in self.blocks:
            block.log.reset()



if __name__ == '__main__':
    import args
    model = QStack(args.get_debug_args())
    model.eval()

    x= torch.FloatTensor(16, 3, 128, 128).normal_()
    outs = model.all_levels_recon(x)
    new  = model.all_levels_recon_new(x)
