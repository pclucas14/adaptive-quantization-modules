import utils
import torch
from torch import nn
from copy import deepcopy
from torch.nn import functional as F

from buffer import *
from vqvae  import Quantize, GumbelQuantize, AQuantize, Encoder, Decoder

# Quantization Building Block
# ------------------------------------------------------

class QLayer(nn.Module):
    def __init__(self, id, args):
        super().__init__()

        self.id      = id
        self.args    = args
        self.log     = utils.RALog()

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
                            decay=args.decay, size=size)]
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
            self.n_samples = 0
            self.n_memory  = 0
            self.mem_per_sample = 0
            self.comp_rate   = args.comp_rate

            buffers = []
            for shp in self.args.argmin_shapes:
                buffers += [Buffer(shp, args.n_classes, dtype=torch.LongTensor)]
                self.mem_per_sample += buffers[-1].mem_per_sample

            self.buffer = nn.ModuleList(buffers)

            assert self.mem_per_sample == np.prod(args.data_size) / self.comp_rate


    def add_to_buffer(self, argmins, y, t, idx=None):
        """ adds indices to layer buffer """

        assert len(argmins) == len(self.buffer)
        n_memory = 0

        if idx is None:
            idx = torch.ones_like(y)

        # TODO: MUST ENSURE THAT THE BUFFER SHUFFLING IS THE SAME ACROSS BUFFER
        swap_idx = torch.randperm(int(self.n_samples))[:(idx == 1).sum()]

        for argmin, buffer in zip(argmins, self.buffer):
            # it's possible some samples in the batch were added in full res.
            # we therefore take them out
            argmin = argmin[-y.size(0):]
            buffer.add(argmin[idx], y[idx], t, swap_idx)

            n_memory += buffer.n_memory

        self.n_samples += idx.sum()
        self.n_memory = n_memory


    def rem_from_buffer(self, n_samples):
        """ only adding this header cuz all other methods have one """

        n_memory = 0
        import pdb

        for buffer in self.buffer:
            buffer.free(n_samples)
            n_memory += buffer.n_memory

        self.n_samples -= int(n_samples)
        assert self.n_samples == self.buffer[-1].n_samples \
                == self.buffer[-1].bx.size(0), pdb.set_trace()

        self.n_memory = n_memory


    def sample_from_buffer(self, n_samples):
        """ only adding this header cuz all other methods have one """

        n_samples = int(n_samples)

        idx = torch.randperm(self.n_samples)[:n_samples]

        for i, (qt, buffer) in enumerate(zip(self.quantize, self.buffer)):
            if i == 0:
                out_x = [qt.idx_2_hid(buffer.bx[idx])]
                out_y = buffer.by[idx]
            else:
                out_x += [qt.idx_2_hid(buffer.bx[idx])]
                assert (out_y - buffer.by[idx]).abs().sum() == 0

        return torch.cat(out_x, 1), out_y


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

        '''
        # you have two options here. You can either use as input
        # 1) the output of the stream from the bottom block
        # 2) the output of the quantizer from the same block

        # The code below was removed; better handled in Q_stack
        if kwargs.get('inter_level_stream', False):
            self.output = self.decoder(x)           # option 1
        else:
            self.output = self.decoder(self.z_q)    # option 2
        '''

        self.output = self.decoder(x)

        return self.output


    def loss(self, target, **kwargs):
        """ Loss calculation """

        if kwargs.get('all_levels_recon', False):
            self.output = self.output[-target.size(0):]

        # TODO: should we weight these differently ?
        diffs = sum(self.diffs) / len(self.diffs)
        #recon = F.mse_loss(self.output, target)
        recon = F.l1_loss(self.output, target)

        self.recon = recon.item()
        if not kwargs.get('no_log', False):
            # during eval we actually perform a complete log, so no need for per-task here
            self.log.log('Distill_recon-B%d' % self.id, self.recon, per_task=False)

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
            mem_size -= sum(p.numel() for p in self.parameters())

            self.mem_size = mem_size    # total floats that can be stored across all blocks
            self.n_seen_so_far = 0      # number of samples seen so far
            self.reg_stored = 0         # samples stored in the regular (uncompressed) buffer
            self.all_stored = 0         # samples stored across all blocks
            self.mem_used = 0           # total float used across blocks
            self.data_size = np.prod(args.data_size)

            self.reg_buffer = Buffer(args.data_size, \
                    args.n_classes, dtype=torch.FloatTensor)

            comp_rate = [self.data_size / block.args.comp_rate for block in self.blocks]
            self.register_buffer('mem_per_block', torch.Tensor([self.data_size] + comp_rate))


    def sample_from_buffer(self, n_samples):
        """ something something something """

        # sample proportional to the amounts of points per resolution
        probs = torch.Tensor([self.reg_stored] + [x.n_samples for x in self.blocks])
        probs = probs / probs.sum()
        samples_per_block = (probs * n_samples).floor()

        #TODO: this will throw an error if no samples are stored in full resolution
        samples_per_block[0] = n_samples - samples_per_block[1:].sum()

        reg_x, reg_y = self.reg_buffer.sample(samples_per_block[0])

        if samples_per_block[0] == n_samples:
            return reg_x, reg_y

        # we reverse the blocks, so that all the decoding can be done in one pass
        r_blocks, r_spb = self.blocks[::-1], reversed(samples_per_block[1:])

        i = 0
        out_y = [reg_y]
        for (block_samples, block) in zip(r_spb, r_blocks):
            if block_samples == 0 and i == 0:
                continue

            xx, yy  = block.sample_from_buffer(block_samples)
            out_y += [yy]

            if i == 0:
                out_x = xx
            else:
                out_x = torch.cat((out_x, xx))

            out_x = block.decoder(out_x)

            i += 1

        return torch.cat((out_x, reg_x)), torch.cat(out_y)


    def add_reservoir(self, x, y, t, **kwargs):
        """ Reservoir Sampling Buffer Addition """

        mem_free = self.mem_size - self.mem_used

        # THIS PART ADDS UNCOMPRESSED REPRESENTATION USING THE FREE MEMORY
        # UNCOMMENT ALSO TODO 2 if activating this block
        '''
        can_store_uncompressed = csu = int(min((mem_free) // x[0].numel(), x.size(0)))

        if can_store_uncompressed > 0:
            self.reg_buffer.add(x[:csu], y[:csu], t, swap_idx=None)
            x, y = x[csu:], y[csu:]

            # update statistic
            self.reg_stored += csu
            self.all_stored += csu
            self.n_seen_so_far += csu
            self.mem_used += self.data_size * csu
        '''

        if x.size(0) > 0:
            # in reservoir sampling, samples should be added with
            # p(amt of samples that fit in mem / samples see so far)
            indices = torch.FloatTensor(x.size(0)).to(x.device).\
                    uniform_(0, self.n_seen_so_far).long()
            valid_indices = (indices < self.all_stored).long()

            # indices of samples to be added in mem
            # note that this process is independant of the compression rate
            # which should make things less biased.
            idx_new_data = valid_indices.nonzero().squeeze(-1)

            # now that we know which samples will be added, we need to check
            # which rep / compression rate will be used.

            """ only using recon error for now """
            target = self.all_levels_recon[:, :x.size(0)]

            # now that we know which samples will be added to the buffer,
            # we need to find the most compressed representation that is good enough
            per_block_l1 = (x.unsqueeze(0) - target).abs()
            per_block_l1 = per_block_l1.mean(dim=(2,3,4))
            block_id = (per_block_l1 < self.args.recon_th).sum(dim=0)

            # we calculate the amount of space that needs to be freed
            space_needed = F.one_hot(block_id, len(self.blocks) + 1).float()
            space_needed = (space_needed * self.mem_per_block).sum()

            # remove the one that's already free
            # TODO 2
            space_needed = (space_needed - mem_free).clamp_(min=0.)

            # we want the removal of samples in the buffer to be agnostic to the
            # compression rate. We determine how much to remove from every block
            # E[removed from b_i] = space_bi_takes / total_space * space_to_be_removed
            to_be_removed_weights = torch.Tensor([self.mem_used] + [x.n_memory for x in self.blocks])
            to_be_removed_weights = to_be_removed_weights / to_be_removed_weights.sum()

            tbr_per_block_mem = to_be_removed_weights * space_needed
            tbr_per_block_n_samples = (tbr_per_block_mem / self.mem_per_block.cpu()).ceil()

            # if nothing needs to be deleted, this tensor will be `nan`. This is a simple fix
            tbr_per_block_n_samples[tbr_per_block_n_samples != tbr_per_block_n_samples] = 0.

            # finally, we iterate over the blocks and add / remove the required samples
            # 0th block (uncompressed)
            self.reg_buffer.free(tbr_per_block_n_samples[0])
            self.reg_buffer.add(x[block_id == 0], y[block_id == 0], t, swap_idx=None)

            # update statistics
            delta_reg = int((block_id == 0).sum() - tbr_per_block_n_samples[0])
            self.reg_stored += delta_reg
            self.all_stored += delta_reg
            self.mem_used   += delta_reg * self.data_size

            for i, block in enumerate(self.blocks):
                # free space
                block.rem_from_buffer(tbr_per_block_n_samples[i + 1])

                # add new points
                block.add_to_buffer(block.argmins, y, t, idx=(block_id == (i+1)))

                # update statistics
                sample_delta = int((block_id == (i+1)).sum() - tbr_per_block_n_samples[i + 1])
                self.all_stored += sample_delta
                self.mem_used   += sample_delta * block.mem_per_sample

            """ Making sure everything is behaving as expected """

            # update statistic
            self.n_seen_so_far += idx_new_data.size(0)

            samples_stored, mem_used = 0, 0
            samples_stored += self.reg_stored
            mem_used += self.reg_stored * np.prod(self.args.data_size)
            samples_in_block = [self.reg_stored]

            for block in self.blocks:
                samples_stored += block.n_samples
                mem_used += block.n_memory
                samples_in_block += [block.n_samples]
                block.log.log('buffer-samples-B%d' % block.id, block.n_samples, per_task=False)

            block.log.log('buffer_samples-reg', self.reg_stored, per_task=False)
            block.log.log('buffer-mem', self.mem_used, per_task=False)

            import pdb
            assert samples_stored == self.all_stored, pdb.set_trace()
            assert mem_used == self.mem_used

            '''
            print('samples per block ', samples_in_block)
            print('mem_used ', mem_used)
            print('samples stored ', samples_stored, '\n')
            '''


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
                loss.backward(retain_graph=(i+1) != len(self.blocks))

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


    def memory_consolidation(self, x_recon, y, t, err):
        """ Adds data from the new task in the buffers """


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
            block.log.log('Full_recon-B%d' % block.id, F.mse_loss(x, og))

        return out_levels


    def log(self, task, writer=None, should_print=False, mode='train'):
        """ Logs the results """

        if writer is not None:
            for block in self.blocks:
                for name, value in block.log.storage.items():
                    prefix = mode + '/'
                    if block.log.per_task[name]:
                        suffix = '__task' + str(task)
                    else:
                        suffix = ''
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


    def calc_grad_var(self, input, **kwargs):
        """ Estimate the variance of the gradient along the batch axis """

        assert self.args.optimization == 'blockwise', 'No global opt for now'

        for i, input_i in enumerate(input):
            # We have to go 1 example at a time, as gradient is acc. over
            # batch axis by default
            input_i   = input_i.unsqueeze(0)
            x_prime_i = self.forward(input_i)

            for j, block in enumerate(self.blocks):
                target_j = input_i if j == 0 else self.blocks[j-1].z_q
                recon, diff = block.loss(target_j.detach(), **kwargs)
                loss = recon + block.args.commitment_cost * diff

                block.opt.zero_grad()
                loss.backward()

                per_block_sum, per_block_cnt = 0., 0.
                for p in list(filter(lambda p: p.grad is not None, block.parameters())):
                    if i == 0:
                        p.grad_mu  = deepcopy(p.grad.data)
                        p.grad_std = deepcopy(p.grad.data).data.fill_(0.)
                    else:
                        prev_mu = deepcopy(p.grad_mu)
                        p.grad_mu.data.add_ ( (p.grad.data - p.grad_mu) / (i+1) )
                        p.grad_std.data.add_( (p.grad.data - prev_mu) * (p.grad.data - p.grad_mu))

                    if (i + 1) == input.size(0):
                        # last example --> log
                        per_block_sum += p.grad_std.sum()
                        per_block_cnt += p.grad_std.numel()

                if (i + 1) == input.size(0):
                    block.log.log('grad_var-B%d' % block.id, per_block_sum / per_block_cnt * 1000., per_task=False)

# Classifiers
# -----------------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, compressed = False):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.compressed = compressed

        self.conv1 = nn.Conv2d(3, nf, kernel_size=7, stride=3,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x):
        bsz = x.size(0)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, 3, 128, 128))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.return_hidden(x)
        out = self.linear(out)
        return out


def ResNet18(nclasses, nf=100, compressed=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, compressed=compressed)


if __name__ == '__main__':
    import args
    model = QStack(args.get_debug_args())
    model.eval()

    x= torch.FloatTensor(16, 3, 128, 128).normal_()
    outs = model.all_levels_recon(x)
    new  = model.all_levels_recon_new(x)
