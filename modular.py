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

        # you have two options here. You can either use as input
        # 1) the output of the stream from the bottom block
        # 2) the output of the quantizer from the same block

        '''
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
            self.reg_buffer = Buffer(args.mem_size, args.data_size, \
                    args.n_classes, dtype=torch.FloatTensor)


    # madd design pattern use. Vybihal and Shashi would be proud
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

        for i, block in enumerate(reversed(self.blocks)):
            if kwargs.get('inter_level_stream', False):
                input = x                                   # option 2
            else:
                if kwargs.get('all_levels_recon', False):
                    # removing deepest block as x == block.z_q for it
                    if i == 0:
                        input = x
                    else:
                        x = x.detach()
                        input = torch.cat((x, block.z_q))   # option 3
                else:
                    input = block.z_q                       # option 1

            x = block.down(input, **kwargs)

        # if `all_levels_recon`, returns a tensor of shape
        # (bs * n_levels, C, H, W). can call `.view(bs, n_levels, ...)
        # to split correctly. Levels are ordered from deepest (top) to bot.
        # i.e. the last one will have the best reconstruction

        return x


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


    def init_buffers(self):
        """ create buffers used for experience replay """
        pass


    def add_reservoir(self, x, y, t):
        """ Reservoir Sampling Buffer Addition """

        if self.rehearsal_level == -1:
            self.reg_buffer.add_reservoir(x, y, t)
            return self.reg_buffer.place_left
        else:
            ''' assumes `x` has just been push through the network,
                and that the saved `z_q` will match x               '''
            self.blocks[self.rehearsal_level].add_reservoir(y, t)
            return self.blocks.buffers[0].place_left



    def memory_consolidation(self, x_recon, y, t, err):
        """ Adds data from the new task in the buffers """


    def all_levels_recon(self, og, **kwargs):
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


if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    import args
    import pdb; pdb.set_trace()
    model = QStack(args.get_debug_args())
    model.eval()

    x= torch.FloatTensor(16, 3, 128, 128).normal_()
    outs = model.all_levels_recon(x)
    new  = model.all_levels_recon_new(x)

    '''
    block = model.blocks[0]
    idx = torch.FloatTensor(17, 64, 64).uniform_(0, 21).long()
    xx = block.idx_2_hid([idx])

    idx = [[idx], [idx[:, :16, :16], idx[:, :32, :32]]]
    outs = model.decode_indices(idx[::-1])
    import pdb; pdb.set_trace()
    xx = 1
    '''
