import utils
import torch
from torch import nn
from torch.nn import functional as F

from vqvae import Quantize, GumbelQuantize, AQuantize, Encoder, Decoder

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
                            args.num_embeddings // size ** 2,
                            decay=args.decay, size=size)]
            elif args.model == 'gumbel':
                qt += [GumbelQuantize(args.embed_dim // args.num_codebooks)]
            elif args.model == 'argmax':
                qt += [AQuantize(args.embed_dim // args.num_codebooks)]

        self.quantize = nn.ModuleList(qt)

        if args.optimization == 'blockwise':
            # build layer opt
            self.opt = torch.optim.Adam(self.parameters(), lr=args.learning_rate)

        # loss specific parameters
        self.register_parameter('dec_log_stdv', \
                torch.nn.Parameter(torch.Tensor([0.])))


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

        # store as scalars for conveniance
        for i in range(len(self.diffs)):
            self.log.log('%d-ppl_%d'  % (self.id, i), self.ppls[i])
            self.log.log('%d-diff_%d' % (self.id, i), self.diffs[i])

        return z_q


    def down(self, x, **kwargs):
        """ Decoding Process """

        # you have two options here. You can either use as input
        # 1) the output of the stream from the bottom block
        # 2) the output of the quantizer from the same block

        if kwargs.get('inter_level_stream', False):
            self.output = self.decoder(x)           # option 1
        else:
            self.output = self.decoder(self.z_q)    # option 2

        return self.output


    def loss(self, target):
        """ Loss calculation """

        # TODO: should we weight these differently ?
        diffs = sum(self.diffs) / len(self.diffs)
        recon = F.mse_loss(self.output, target)

        self.recon = recon.item()
        self.log.log('%d-recon' % self.id, self.recon)

        return recon, diffs


# Quantization Network (stack QLayers)
# ------------------------------------------------------

class QStack(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.log_step = 0

        """ assumes args is a nested dictionary, one for every block """
        blocks = []
        for layer_no in range(len(args.layers)):
            blocks += [QLayer(layer_no, args.layers[layer_no])]

        self.blocks = nn.ModuleList(blocks)

        if self.args.optimization == 'global':
            self.opt = torch.optim.Adam(self.parameters(), \
                    lr=args.global_learning_rate)


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

        for block in reversed(self.blocks):
            if kwargs.get('inter_level_stream', False):
                input = x           # option 2
            else:
                input = block.z_q   # option 1

            x = block.down(input, **kwargs)

        return x


    def forward(self, x, **kwargs):
        x = self.up(x,   **kwargs)
        x = self.down(x, **kwargs)
        return x


    def optimize(self, target, **kwargs):
        """ Loss calculation """

        total_loss = 0.
        for i, block in enumerate(self.blocks):
            # for now, let's fix the target of the subblocks to be `z_e` and not `z_q`
            target_i = target if i == 0 else self.blocks[i-1].z_q# z_e

            # it's important to detach the target! (similar to RL / Q-learning)
            recon, diff = block.loss(target_i.detach())

            # TODO: figure out a way to train only on final recon
            # TODO: put this back
            # loss = (0. if i > 0 else 1.) * recon + block.args.commitment_cost * diff
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


    def all_levels_recon(self, og):
        """ Expand all levels to input space to evaluate reconstruction """

        # encode
        kwargs = {'inter_level_stream' : True, 'stop':True}
        self.up(og, **kwargs)

        out_levels = []
        for i, block in enumerate(reversed(self.blocks)):
            # current block
            x = block.down(block.z_q, **kwargs)
            for j, block_ in enumerate(self.blocks[::-1][i+1:]):
                x = block_.down(x, **kwargs)

            # store output
            out_levels += [x]

            # log reconstruction error
            block.log.log('%d-recon' % block.id, F.mse_loss(x, og))

        return out_levels


    def log(self, writer=None, should_print=False, prefix=''):
        """ Logs the results """

        if writer is not None:
            for block in self.blocks:
                for name, value in block.log.storage.items():
                    writer.add_scalar(prefix + ' ' + name, value, self.log_step)

            self.log_step += 1

        if should_print:
            print(prefix)
            for block in self.blocks:
                print(block.log.one_liner())
            print('\n')

        # reset logs
        for block in self.blocks:
            block.log.reset()
