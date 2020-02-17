import os
import torch
from torch import nn
from copy import deepcopy
from torch.nn import functional as F

from utils.utils     import *
from utils.buffer    import *
from common.quantize import Quantize, GumbelQuantize, AQuantize, CQuantize, SoftQuantize
from common.model    import Encoder, Decoder, ResNet18

from utils.kitti_utils import *

from torchvision.utils import save_image
from PIL import Image


# Quantization Building Block
# ------------------------------------------------------

class QLayer(nn.Module):
    def __init__(self, id, args):
        super().__init__()

        self.id        = id
        self.args      = args
        self.log       = RALog()
        self.avg_comp  = 0.

        if args.downsample > 1:
            # build networks
            self.encoder = Encoder(args)
            self.decoder = Decoder(args)
        else:
            self.encoder = self.decoder = lambda x : x

        # build quantization blocks
        qt    = []
        dtype = torch.LongTensor

        D, K, N = args.embed_dim, args.num_embeddings, args.num_codebooks
        self.quantize = Quantize(D // N, K, N, decay=args.decay)

        if args.optimization == 'blockwise':
            try:
                # build layer opt
                self.opt = torch.optim.Adam(self.parameters(),
                        lr=args.learning_rate)
            except:
                self.opt = None

        self.comp_rate   = args.comp_rate

        # whether or not embedding matrix is frozen
        self.frozen_qt   = False

        argmin_shp = (self.args.num_codebooks, ) + tuple(self.args.argmin_shapes[0])
        assert sum([x != self.args.argmin_shapes[0] for x in self.args.argmin_shapes[1:]]) == 0, pdb.set_trace()
        self.buffer = Buffer(argmin_shp, args.n_classes, max_idx=args.num_embeddings)
        self.mem_per_sample = self.buffer.mem_per_sample

        # same for now
        self.old_quantize = self.quantize
        self.ema_decoder  = self.decoder

        assert -.01 < (self.buffer.mem_per_sample - np.prod(args.data_size) / self.comp_rate) < .01


    @property
    def n_samples(self):
        return self.buffer.n_samples


    @property
    def n_memory(self):
        return self.buffer.n_memory


    def update_ema_decoder(self):
        if not self.frozen_qt or not self.args.use_ema:
            return

        decay = .99
        try:
            for ema_param, param in zip(self.ema_decoder.parameters(), self.decoder.parameters()):
                ema_param.data.copy_(decay * ema_param.data + (1. - decay) * param.data)
        except:
            pass


    def init_ema(self):
        if self.args.use_ema:
            self.ema_decoder = deepcopy(self.decoder)


    def add_to_buffer(self, argmins, y, t, ds_idx, idx=None, step=0):
        """ adds indices to layer buffer """

        if idx is None:
            idx = torch.ones_like(y).bool()

#        assert 'bool' in idx.type().lower()

        #NOTE: we have : data_x  = torch.cat((input_x, re_x))
        argmins = argmins[:y.size(0)]
        self.buffer.add(argmins[idx], y[idx], t, ds_idx[idx], step)


    def rem_from_buffer(self, n_samples=None, idx=None):
        """ only adding this header cuz all other methods have one """

        if n_samples == 0:   return
        if n_samples is None and idx.size(0) == 0: return

        if idx is not None:
            assert n_samples is None
            n_samples = idx.size(0)

        import pdb

        # for buffer in self.buffer:
        self.buffer.free(n_samples, idx=idx)



    def tag_as_compressible(self, n_samples=None, idx=None):
        """ only adding this header cuz all other methods have one """

        import pdb

        for buffer in self.buffer:
            buffer.free(n_samples, idx=idx)

        if idx is not None:
            assert n_samples is None
            n_samples = idx.size(0)

        assert self.n_samples == self.buffer[0].n_samples \
                == self.buffer[-1].bx.size(0), pdb.set_trace()


    def add_argmins(self, y, t, ds_idx, step, argmin_idx=None, last_n=None):
        """ adds new representations to the buffer.
            made to be used with `update_buffer_idx` """

        # note: these arguments should always be passed, at least for now
        assert argmin_idx is not None and last_n is not None

        argmin, buffer = self.argmins, self.buffer
        if last_n is not None:
            argmin = argmin[-last_n:]
        if argmin_idx is not None:
            argmin = argmin[argmin_idx]

        buffer.add(argmin, y, t, ds_idx, step)


    def update_buffer(self, buffer_idx, argmin_idx=None, last_n=None):
        """ update the latent indices stored in the buffer """

        # note: these arguments should always be passed, at least for now
        assert argmin_idx is not None and last_n is not None

        argmin, buffer = self.argmins, self.buffer
        if last_n is not None:
            argmin = argmin[-last_n:]
        if argmin_idx is not None:
            argmin = argmin[argmin_idx]

        buffer.update(buffer_idx, argmin)


    def sample_from_buffer(self, n_samples):
        """ only adding this header cuz all other methods have one """

        with torch.no_grad():
            n_samples = int(n_samples)

            idx = torch.randperm(self.n_samples)[:n_samples]
            self.sampled_indices = idx

            import pdb

            # store the last sampled indices for a potential update
            self.sampled_idx = idx

            qt, buffer  = self.old_quantize, self.buffer

            return qt.idx_2_hid(buffer.bx[idx]), buffer.by[idx], buffer.bt[idx], \
                    buffer.bidx[idx], buffer.bx[idx], buffer.bstep[idx]


    def sample_EVERYTHING(self, **kwargs):
        """ only adding this header cuz all other methods have one """

        BS = 32
        with torch.no_grad():

            n_batches = self.n_samples // BS
            if self.n_samples != n_batches * BS : n_batches += 1

            for batch in range(n_batches):
                # idx = torch.randperm(self.n_samples)
                idx = range(batch * BS, min(self.n_samples, (batch+1) * BS))
                qt, buffer  = self.old_quantize, self.buffer

                to_be_returned = [buffer.by[idx], buffer.bx[idx], buffer.bt[idx], buffer.bidx[idx]]

                if kwargs.get('device', False):
                    dev = kwargs['device']
                    to_be_returned = [item.to(dev) for item in to_be_returned]

                to_be_returned = [qt.idx_2_hid(to_be_returned[1])] + to_be_returned

                yield to_be_returned


    def up(self, x, **kwargs):
        """ Encoding process """

        self.input = x

        # 1) encode
        z_e   = self.encoder(x)

        z_q, diff, argmin, ppl = self.quantize(z_e)

        # Used to be 75 --> now 90 --> now 95
        if self.avg_comp > .90 and not self.frozen_qt and self.args.freeze_embeddings:
            print('fixing Block %d' % self.id)
            self.quantize.decay = 1.
            self.frozen_qt = True
            self.init_ema()
            kwargs['%d_frozen' % self.id] = True


        self.log.log('decay-%d' % self.id, self.quantize.decay, per_task=False)


        # save tensors required tensors for later
        self.z_e     = z_e
        self.z_q     = z_q
        self.ppls    = ppl
        self.diffs   = diff
        self.argmins = argmin

        if not kwargs.get('no_log', False):
            # store as scalars for convenience
            self.log.log('ppl-B%d'  % self.id, self.ppls,  per_task=False)
            self.log.log('diff-B%d' % self.id, self.diffs, per_task=False)

        return z_q


    def down(self, x, **kwargs):
        """ Decoding Process """

        if kwargs.get('ema_decoder', False):
            return self.ema_decoder(x)

        self.output = self.decoder(x)

        return self.output


    def loss(self, target, **kwargs):
        """ Loss calculation """

        if kwargs.get('all_levels_recon', False):
            self.output = self.output[-target.size(0):]

        diffs = self.diffs

        if 'kitti' in self.args.dataset:
            # recon = F.l1_loss(self.output, target)
            recon = F.l1_loss(self.output, target)

        else:
            recon = F.mse_loss(self.output, target, reduction='none')
            if 'next_frame' in kwargs and self.id == 0:
                next_frame = kwargs['next_frame']
                inc_target, re_target = target[:next_frame.size(0)], target[next_frame.size(0):]

                changed_a_lot = (next_frame - inc_target).abs() > 1e-5

                # penalize things that change more
                idx = slice(0, next_frame.size(0))

                lam = .99

                recon[idx] = recon[idx] * lam * changed_a_lot.float() + recon[idx]* (1 - lam) * (1.-changed_a_lot.float())

            recon = recon.mean()

        self.recon = recon.item()
        if not kwargs.get('no_log', False):
            # during eval we actually perform a complete log, so no need for per-task here
            self.log.log('Distill-B%d' % self.id, self.recon, per_task=False)
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
        self.register_buffer('avg_l2',   torch.Tensor([0] * args.num_blocks))

        """ assumes args is a nested dictionary, one for every block """
        blocks = []
        for layer_no in range(len(args.layers)):
            blocks += [QLayer(layer_no, args.layers[layer_no])]

        self.blocks = nn.ModuleList(blocks)

        if args.optimization == 'global':
            self.opt = torch.optim.Adam(self.parameters(), \
                    lr=args.global_learning_rate)

        if True: #args.rehearsal:
            mem_size  = args.mem_size * np.prod(args.data_size)

            if not args.sunk_cost:
                mem_size -= sum(p.numel() for p in self.parameters())
                assert mem_size > 0, 'model is too big to store any samples'

            self.mem_size = mem_size    # total floats that can be stored across all blocks
            self.n_seen_so_far = 0      # number of samples seen so far

            self.data_size = np.prod(args.data_size)
            self.can_store_reg = mem_size // (self.data_size)

            # whether we need to recompute the buffer statistics
            self.up_to_date_mu = self.up_to_date_as  = False

            self.reg_buffer = Buffer(args.data_size, \
                    args.n_classes, dtype=torch.FloatTensor)

            comp_rate     = [block.args.comp_rate for block in self.blocks]
            mem_per_block = [self.data_size / block.args.comp_rate for block in self.blocks]

            self.register_buffer('mem_per_block', torch.Tensor([self.data_size] + mem_per_block))
            self.register_buffer('comp_rate', torch.Tensor(comp_rate + [1.]))

    @property
    def capacity(self):
        current_total = 0
        capacity      = 0

        for block in reversed(self.blocks):
            block_capacity = max(0, block.avg_comp - current_total)
            capacity += block_capacity * block.args.comp_rate
            current_total += block_capacity

        capacity *= self.can_store_reg

        # finally, add the uncompressed samples
        capacity += max(0, 1 - current_total) * self.can_store_reg

        return capacity


    @property
    def all_stored(self):
        total = self.reg_buffer.n_samples

        for block in self.blocks:
            total += block.n_samples

        self.all_stored_ = total
        self.up_to_date_as = True

        return total


    @property
    def mem_used(self):
        total = self.reg_buffer.n_memory

        for block in self.blocks:
            total += block.n_memory

        return total


    @property
    def reg_stored(self):
        return self.reg_buffer.n_samples


    def up_to_date(self, value):
        self.up_to_date_mu = self.up_to_date_as = value


    def update_ema_decoder(self):
        """ update the `old decoders` copy for every block """

        for block in self.blocks:
            block.update_ema_decoder()

    def cut_lr(self):
        for block in self.blocks:
            block.args.decay = block.args.decay + (1. - block.args.decay) * .1
            if block.opt is not None:
                for param_group in block.opt.param_groups:
                    param_group['lr'] = max(param_group['lr'] / 1.3, 1e-5)
                    print('new lr {:.6f}, decay {:.4f}'.format(param_group['lr'], block.args.decay))


    def buffer_update_idx(self, re_x, re_y, re_t, re_ds_idx, re_step, **kwargs):
        # If more than 5% of memory is already free, ski this step
        if float(self.mem_used) / self.mem_size < .95 and not self.args.always_compress: # instead of 90
            ratio = float(self.mem_used) / self.mem_size
            # print('currently at {:.4f}'.format(ratio))
            return

        with torch.no_grad():
            re_target = self.all_levels_recon[:, -re_x.size(0):]

            per_block_l2 = (re_x.unsqueeze(0) - re_target).pow(2)
            per_block_l2 = per_block_l2.mean(dim=(2,3,4))

            if self.args.mask_unfrozen:
                frozen = torch.Tensor([block.frozen_qt for block in self.blocks])
                frozen = frozen.view(-1, 1).expand_as(per_block_l2).to(per_block_l2.device)

                # reconstruction thresholds are ordered from most compressed to least compressed
                frozen = frozen.flip(0)

                super_high_value = torch.ones_like(frozen).fill_(1e9)
                per_block_l2 = per_block_l2 * frozen + (1 - frozen) * super_high_value

            recon_th = self.recon_th.unsqueeze(1).expand_as(per_block_l2)

            # TODO: build block id when sampling
            pre_block_id = self.last_block_id.to(per_block_l2.device)

            ''' '''
            # we want the reconstruction id to be `eps` if the image
            # has already been compressed
            already_compressed = ac = (pre_block_id > 0)
            ac = ac.view(1, -1).expand_as(recon_th).float()

            recon_th = self.args.eps_th * ac + (1 - ac) * recon_th
            block_id = (per_block_l2 < recon_th)
            block_id = block_id.cumsum(dim=0).clamp_(max=1)
            block_id = block_id.sum(dim=0)
            ''' '''

            # take the most compressed rep (biggest block id)
            new_block_id = torch.stack((block_id, pre_block_id)).max(dim=0)[0]

            # first, delete points from real buffer which will be compressed
            delete_idx = self.sampled_indices[(pre_block_id == 0) * (new_block_id > 0)]
            self.reg_buffer.free(idx=delete_idx)

            # monitor what happens
            same_lvl_amt = (pre_block_id == new_block_id).sum()
            jump01_amt   = ((pre_block_id == 0) * (new_block_id == 1)).sum()
            jump12_amt   = ((pre_block_id == 1) * (new_block_id == 2)).sum()

            if self.blocks[0].frozen_qt:
                jump = jump01_amt + jump12_amt
                # print('jump {} / {}'.format(jump.item(), re_x.size(0)))

            self.blocks[0].log.log('same_lvl', same_lvl_amt, per_task=False)
            self.blocks[0].log.log('jump01', jump01_amt, per_task=False)

            for i, block in enumerate(self.blocks):

                if not self.args.no_idx_update:
                    # 1) update stale representations
                    update_mask = (pre_block_id == (i+1)) * (new_block_id == (i+1))
                    update_idx  = self.sampled_indices[update_mask]
                    block.update_buffer(update_idx, argmin_idx=update_mask, last_n=re_x.size(0))
                if False: #Eself.args.delayed_delete:
                    delete_mask = (pre_block_id == (i+1)) * (new_block_id > (i+1))
                    delete_idx  = self.sampled_indices[delete_mask]
                    block.tag_as_compressible(delete_idx)
                    pass
                else:
                    # 2) delete representations that will be further compressed
                    delete_mask = (pre_block_id == (i+1)) * (new_block_id > (i+1))
                    delete_idx  = self.sampled_indices[delete_mask]
                    block.rem_from_buffer(idx=delete_idx)

                    # 3) add new representations
                    add_mask    = (pre_block_id < (i+1))  * (new_block_id == (i+1))
                    block.add_argmins(re_y[add_mask], re_t[add_mask], re_ds_idx[add_mask], re_step[add_mask], argmin_idx=add_mask, last_n=re_x.size(0))

                    if kwargs.get('cpu_buffer', False):
                        cpu_buffer = kwargs['cpu_buffer']
                        cpu_buffer.blocks[i].argmins = block.argmins.cpu()
                        cpu_buffer.blocks[i].add_argmins(re_y[add_mask].cpu(),
                                                         re_t[add_mask].cpu(),
                                                         re_ds_idx[add_mask].cpu(),
                                                         re_step[add_mask].cpu(),
                                                         argmin_idx=add_mask.cpu(),
                                                         last_n=re_x.size(0))


            # will need to recompute statistics
            self.up_to_date(False)


    def sample_from_buffer(self, n_samples, exclude_task=None):
        """ something something something """
        with torch.no_grad():

            # note: the block ids are off by one with block.id :/ i.e. block.id == 0 will be 1 here
            self.block_id = torch.zeros(n_samples).long()

            # sample proportional to the amounts of points per resolution
            probs = torch.Tensor([self.reg_stored] + [x.n_samples for x in self.blocks])

            if self.training:
                # sample proportional to the memory usage. Used in training to maximize space
                # TODO: put this back
                probs = torch.Tensor([self.reg_buffer.n_memory] + [b.n_memory for b in self.blocks])

            probs = probs / probs.sum()
            samples_per_block = (probs * n_samples).floor()

            # make sure we don't sample more than available
            avail = torch.Tensor([self.reg_buffer.n_samples] + [x.n_samples for x in self.blocks])
            samples_per_block = torch.stack([avail, samples_per_block]).min(dim=0)[0]

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

            reg_x, reg_y, reg_t, reg_ds_idx, reg_step =  self.reg_buffer.sample(samples_per_block[0], exclude_task=exclude_task)

            # keep strack of the sampled indices
            self.sampled_indices = []

            if samples_per_block[0] == n_samples:
                self.sampled_indices = self.reg_buffer.sampled_indices
                return reg_x, reg_y, reg_t, reg_ds_idx, reg_ds_idx

            # we reverse the blocks, so that all the decoding can be done in one pass
            r_blocks, r_spb = self.blocks[::-1], reversed(samples_per_block[1:])

            i = 0
            for (block_samples, block) in zip(r_spb, r_blocks):
                if block_samples == 0 and i == 0:
                    continue

                xx, yy, tt, ds_idx, _, step  = block.sample_from_buffer(block_samples)

                self.sampled_indices = [block.sampled_indices] + self.sampled_indices

                if i == 0:
                    out_x = xx
                    out_y = yy
                    out_t = tt
                    out_idx = ds_idx
                    out_step = step
                else:
                    out_x = torch.cat((xx, out_x))
                    out_y = torch.cat((yy, out_y))
                    out_t = torch.cat((tt, out_t))
                    out_idx = torch.cat((ds_idx, out_idx))
                    out_step = torch.cat((step, out_step))

                # use old weights when sampling
                out_x = block.ema_decoder(out_x)

                i += 1

            # TODO: check if should be on CUDA already
            self.sampled_indices = torch.cat([self.reg_buffer.sampled_indices.cpu()] + \
                    self.sampled_indices)

            return torch.cat((reg_x, out_x)), torch.cat((reg_y, out_y)), torch.cat((reg_t, out_t)), torch.cat((reg_ds_idx, out_idx)), torch.cat((reg_step, out_step))


    def sample_EVERYTHING(self, **kwargs):
        """ something something something """
        with torch.no_grad():

            reg_x, reg_y, reg_t, reg_idx = self.reg_buffer.bx, self.reg_buffer.by, \
                                           self.reg_buffer.bt, self.reg_buffer.bidx

            n_batches = reg_x.size(0) // 128
            if n_batches != reg_x.size(0) * 128: n_batches += 1

            for i in range(n_batches):
                idx = range(i * 128, min(reg_x.size(0), (i+1) * 128))
                yield reg_x[idx], reg_y[idx], None, reg_t[idx], reg_idx[idx], -1


            # we reverse the blocks, so that all the decoding can be done in one pass
            r_blocks = self.blocks[::-1]

            i = 0
            for block in r_blocks:

                try:
                    gen_iter = block.sample_EVERYTHING(**kwargs)
                    while True:
                        xx, yy, argmin, tt, ds_idx  = next(gen_iter)

                        xx = block.ema_decoder(xx)
                        for j, block_ in enumerate(self.blocks[::-1][i+1:]):
                            xx = block_.ema_decoder(xx)

                        yield xx, yy, argmin, tt, ds_idx, block.id

                except StopIteration:
                    i += 1


    def add_to_buffer(self, x, y, t, ds_idx, step=0, **kwargs):
        with torch.no_grad():

            # first we add all incoming points
            target = self.all_levels_recon[:, :x.size(0)]

            # now that we know which samples will be added to the buffer,
            # we need to find the most compressed representation that is good enough

            per_block_l2 = (x.unsqueeze(0) - target).pow(2)
            per_block_l2 = per_block_l2.mean(dim=(2,3,4))
            self.avg_l2  = self.avg_l2   * 0.99 + .01 * per_block_l2.mean(-1).flip(0)

            recon_th  = self.recon_th.unsqueeze(1).expand_as(per_block_l2)
            block_id  = (per_block_l2 < recon_th)
            comp_rate = block_id.float().mean(dim=1).flip(0)

            if self.args.mask_unfrozen:
                frozen = torch.Tensor([block.frozen_qt for block in self.blocks])
                frozen = frozen.view(-1, 1).expand_as(per_block_l2).to(per_block_l2.device)

                # reconstruction thresholds are ordered from most compressed to least compressed
                frozen = frozen.flip(0)

                super_high_value = torch.ones_like(frozen).fill_(1e9)
                per_block_l2 = per_block_l2 * frozen + (1 - frozen) * super_high_value

            block_id  = (per_block_l2 < recon_th)

            # say block 2 fits its criteria, but not block 1. make sure the sum to
            # come does not give you 1
            block_id = block_id.cumsum(dim=0).clamp_(max=1)
            block_id = block_id.sum(dim=0)

            # block_ids are off-by-1: block_id == 0 means regular buffer, not block 0
            # let's shift everything so that block_id == 0 means block 0
            block_id = block_id - 1

            # now that we know which samples goes where, we can add them
            # self.reg_buffer.add(x[block_id == 0], y[block_id == 0], t, ds_idx[block_id == 0],  step)
            self.reg_buffer.add(x[block_id == -1], y[block_id == -1], t, ds_idx[block_id == -1],  step)

            for i, block in enumerate(self.blocks):
                # add new points
                # block.add_to_buffer(block.argmins, y, t, ds_idx, idx=(block_id == (i+1)), step=step)
                block.add_to_buffer(block.argmins, y, t, ds_idx, idx=(block_id == i), step=step)

                if kwargs.get('cpu_buffer', False):
                    cpu_buffer = kwargs['cpu_buffer']
                    cpu_buffer.blocks[i].add_to_buffer(block.argmins.cpu(), y.cpu(), t, ds_idx.cpu(), idx=(block_id == i).cpu(), step=step)

                # log
                block.avg_comp = block.avg_comp * 0.99 + .01 * comp_rate[i].item()
                block.log.log('buffer-%d-comp_rate' % block.id, block.avg_comp, per_task=False)
                block.log.log('buffer-%d-avg_l2'    % block.id, self.avg_l2[i].item(), per_task=False)


            # now that points are added, we may need to remove some in order to fit in memory
            should_free = self.mem_used - self.mem_size

            if should_free > 0:
                # TODO: put this back!
                steps = torch.cat([self.reg_buffer.bstep] + [x.buffer.bstep for x in self.blocks])

                # steps = torch.cat([self.reg_buffer.by] + [x.buffer.by for x in self.blocks])
                ids   = torch.cat([torch.LongTensor(x.n_samples).fill_(x.id) for x in self.blocks]).to(steps.device)
                block_specific_idx = torch.cat([torch.arange(self.reg_buffer.n_samples)] + [torch.arange(x.n_samples) for x in self.blocks])

                # we use `-1` to denote the uncompressed samples
                ids   = torch.cat((torch.LongTensor(self.reg_buffer.n_samples).fill_(-1).to(ids.device), ids))

                # sort chronologically
                sorted_steps, sorted_idx = steps.sort()
                sorted_ids               = ids[sorted_idx]
                block_specific_idx       = block_specific_idx[sorted_idx]

                WIDTH = 4
                to_odd = lambda x : x+1 if x % 2 == 0 else x
                window_size = min(to_odd(self.can_store_reg), to_odd(self.all_stored))
                missing_per_side = window_size // 2

                window = sorted_steps.unfold(0, int(window_size), 1)
                left   = window[0].unsqueeze(0).expand(missing_per_side, -1)
                right  = window[-1].unsqueeze(0).expand(missing_per_side, -1)
                window = torch.cat((left, window, right))

                delta = (sorted_steps.view(-1, 1) - window).float()
                kernel_input = (-delta ** 2 / (2*WIDTH**2)).exp().div_(WIDTH * np.sqrt(2 * np.pi))

                kernel = kernel_input.mean(1)

                kernel = F.softmax(kernel / 1e-4, dim=-1)

                # I commented it out for exact reproducibility of the Imagenet Exp, but
                # ideally it should be there
                # kernel.clamp_(min=1e-9)
                # kernel = kernel / kernel.sum()

                if (kernel != kernel).sum() > 0:
                    import pdb; pdb.set_trace()
                # kernel = kernel / kernel.sum()

                # sample according to kernel
                best_block_used = ids.max()
                if best_block_used == -1:
                    n_samples = min(np.ceil(should_free / self.reg_buffer.mem_per_sample), self.reg_buffer.n_samples)
                else:
                    n_samples = min(np.ceil(should_free / self.blocks[best_block_used].mem_per_sample), max([x.n_samples for x in self.blocks]))

                import pdb
                assert kernel.size(0) > n_samples, pdb.set_trace()

                if self.n_seen_so_far > 2000 and False:
                    import matplotlib.pyplot as plt
                    from PIL import Image
                    plt.plot(sorted_steps.cpu().numpy(), kernel.cpu().numpy())
                    plt.ylim(0, kernel.max().item())
                    plt.xlim(0, sorted_steps.max().item())
                    plt.title('kernel')
                    plt.savefig('tmpa.png')
                    Image.open('tmpa.png').show()

                    plt.cla()
                    bins = np.linspace(0, steps.max().item(), 100)
                    plt.hist(sorted_steps.cpu().numpy(), bins=bins)
                    plt.savefig('tmpd.png')
                    Image.open('tmpd.png').show()
                    import pdb; pdb.set_trace()

                kernel_np = kernel.cpu().numpy()
                if kernel[kernel != kernel].size(0) > 0:
                    print('nans')
                    import pdb; pdb.set_trace()
                    xx = 1

                # if this fails, it's because not enough categories have non zero prob
                idx_sample = torch.multinomial(kernel, int(n_samples), replacement=False)

                # idx_sample = kernel.topk(int(n_samples))[1]
                id_sample  = sorted_ids[idx_sample]

                # import pdb; pdb.set_trace()
                # print(sorted_steps[idx_sample].float().mean(), sorted_steps.float().mean())

                # all that remains is to subsample this sample prop. to memory usage
                samples_per_block = id_sample.new(id_sample.size(0), len(self.blocks) + 1).fill_(0)
                samples_per_block = samples_per_block.scatter_(1, id_sample.view(-1, 1) + 1, 1).sum(0)
                # [reg_buffer_samples, b0_samples, ..., bn_samples]
                mem_per_block     = samples_per_block * self.mem_per_block
                prob_per_block    = mem_per_block / mem_per_block.sum()
                to_be_sampled     = ((prob_per_block * should_free)  / self.mem_per_block).ceil().long()

                # remove uncompressed
                if to_be_sampled[0] > 0 and not self.args.rl:
                    valid_idx = idx_sample[id_sample == -1]
                    drawn_idx = valid_idx[torch.randperm(valid_idx.size(0))[:to_be_sampled[0]]]
                    drawn_idx = block_specific_idx[drawn_idx]
                    self.reg_buffer.free(idx=drawn_idx)

                for block in self.blocks:
                    i = block.id + 1
                    if to_be_sampled[i] > 0:

                        valid_idx = idx_sample[id_sample == block.id]
                        drawn_idx = valid_idx[torch.randperm(valid_idx.size(0))[:to_be_sampled[i]]]
                        drawn_idx = block_specific_idx[drawn_idx]
                        block.buffer.free(idx=drawn_idx)


            """ Making sure everything is behaving as expected """

            import pdb
            assert self.mem_size >= self.mem_used or self.args.rl, pdb.set_trace()

            # update statistic
            self.n_seen_so_far += x.size(0)

            for block in self.blocks:
                block.log.log('buffer-samples-B%d' % block.id, block.n_samples, per_task=False)

            block.log.log('buffer_samples-reg', self.reg_stored, per_task=False)
            block.log.log('buffer-mem', self.mem_used, per_task=False)
            block.log.log('n_seen_so_far', self.n_seen_so_far, per_task=False)


    def add_reservoir(self, x, y, t, ds_idx, step=0, **kwargs):
        """ Reservoir Sampling Buffer Addition """

        with torch.no_grad():
            mem_free = self.mem_size - self.mem_used

            if x.size(0) > 0:
                # in reservoir sampling, samples should be added with
                # p(amt of samples that fit in mem / samples see so far)
                indices = torch.FloatTensor(x.size(0)).to(x.device).\
                        uniform_(0, self.n_seen_so_far).long()

                capacity = self.all_stored # self.capacity
                valid_indices = (indices < max(self.can_store_reg, capacity)).long()
                self.blocks[0].log.log('capacity', capacity, per_task=False)

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
                self.avg_l2  = self.avg_l2   * 0.99 + .01 * per_block_l2.mean(-1).flip(0)

                recon_th = self.recon_th.unsqueeze(1).expand_as(per_block_l2)
                block_id  = (per_block_l2 < recon_th)
                comp_rate = block_id.float().mean(dim=1).flip(0)


                for i, block in enumerate(self.blocks):
                    block.avg_comp = block.avg_comp * 0.99 + .01 * comp_rate[i].item()
                    block.log.log('buffer-%d-comp_rate' % block.id, block.avg_comp, per_task=False)
                    block.log.log('buffer-%d-avg_l2'    % block.id, self.avg_l2[i].item(), per_task=False)

                if self.args.mask_unfrozen:
                    frozen = torch.Tensor([block.frozen_qt for block in self.blocks])
                    frozen = frozen.view(-1, 1).expand_as(per_block_l2).to(per_block_l2.device)

                    # reconstruction thresholds are ordered from most compressed to least compressed
                    frozen = frozen.flip(0)

                    super_high_value = torch.ones_like(frozen).fill_(1e9)
                    per_block_l2 = per_block_l2 * frozen + (1 - frozen) * super_high_value

                block_id  = (per_block_l2 < recon_th)

                # say block 2 fits its criteria, but not block 1. make sure the sum to
                # come does not give you 1

                block_id = block_id.cumsum(dim=0).clamp_(max=1)
                block_id = block_id.sum(dim=0)

                # on the off chance that all samples fit in memory, add them all
                space_needed = F.one_hot(block_id, len(self.blocks) + 1).float()
                space_needed = (space_needed * self.mem_per_block).sum()
                space_needed = (space_needed - mem_free).clamp_(min=0.)

                # UPDATE: actually adding everything when space allows it causes
                # imbalance in the stream
                # TODO: revert this
                if space_needed == 0 and False:
                    # add all the points
                    idx_new_data = torch.arange(x.size(0))
                else:
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
                self.reg_buffer.add(x[block_id == 0], y[block_id == 0], t, ds_idx[block_id == 0],  step)

                for i, block in enumerate(self.blocks):
                    # free space
                    block.rem_from_buffer(tbr_per_block_n_samples[i + 1])

                    # add new points
                    block.add_to_buffer(block.argmins, y, t, ds_idx, idx=(block_id == (i+1)), step=step)

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
                block_hist = block.buffer.y.sum(dim=0).cpu()
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

        for i, block in enumerate(self.blocks):

            if i > 0:
                if self.args.modular_input == 'z_q':
                    x = last_same_size_zq
                else:
                    x = last_same_size_ze

            if self.args.optimization == 'global':
                x = x               # option 1
            else:
                x = x.detach()      # option 2

            x = block.up(x, **kwargs)

            if i == 0 or block.args.downsample > 1:
                last_same_size_zq = block.z_q
                last_same_size_ze = block.z_e

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
            # removing deepest block as x == block.z_q for it
            if i == 0:
                input = x                           # option 2
            else:
                x = x.detach()
                input = torch.cat((x, block.z_q))   # option 1
                #input = torch.cat((x, block.z_e))   # option 1

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
            # target_i = target if i == 0 else self.blocks[i-1].z_q
            target_i = target if i == 0 else block.input

            #target_i = target if i == 0 else self.blocks[i-1].z_e
            #target_i = target if i == 0 else self.blocks[0].z_q

            # it's important to detach the target! (similar to RL / Q-learning)
            recon, diff = block.loss(target_i.detach(), **kwargs)

            # TODO: make this better


            loss = recon + block.args.commitment_cost * diff

            if self.args.optimization == 'global':
                total_loss += loss
            elif block.opt is not None:
                # optimize
                block.opt.zero_grad()
                loss.backward()
                block.opt.step()

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

        with torch.no_grad():
            # encode
            self.up(og, **kwargs)

            out_levels = []
            for i, block in enumerate(reversed(self.blocks)):
                # current block

                if kwargs.get('ema_decoder', False):
                    x = block.ema_decoder(block.z_q)
                else:
                    x = block.decoder(block.z_q)

                for j, block_ in enumerate(self.blocks[::-1][i+1:]):
                    if kwargs.get('ema_decoder', False):
                        x = block_.ema_decoder(x)
                    else:
                        x = block_.decoder(x)

                # store output
                out_levels += [x]

                name = 'Full_recon-B%d' % block.id
                if kwargs.get('ema_decoder', False): name += '_ema'
                # block.log.log(name, F.mse_loss(x, og))
                block.log.log(name, F.mse_loss(x, og), per_task=True)
                block.log.log(name, F.mse_loss(x, og), per_task=False)

            return out_levels


    def log(self, task, writer=None, should_print=False, mode='train'):
        """ Logs the results """

        # get buffer informations
        if 'kitti' not in self.args.dataset:
            self.log_buffer()

        prefix = None
        if writer is not None:
            for block in self.blocks:
                for name, value in block.log.storage.items():
                    prefix = mode + '/'
                    if block.log.per_task[name]:
                        suffix = '__task' + str(task)
                    else:
                        suffix = ''

                    if type(value) == np.ndarray:
                        if mode == 'buffer':
                            # Tensorboard's histogram is awful. Let's make an image instead
                            tmp_path = os.path.join(self.args.log_dir, 'tmp.png')
                            hist = make_histogram(value, prefix + name + suffix, tmp_path=tmp_path)
                            writer.add_image(prefix + name + suffix, hist, self.log_step)
                    else:
                        writer.add_scalar(prefix + name + suffix, value, self.log_step)

            self.log_step += 1

        if should_print:
            if prefix is not None: print(prefix)
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
