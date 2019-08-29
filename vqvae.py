import utils
"""
Vector Quantization

DiffVQVAE author: Andre Cianflone

Original VQVAE code based on :
https://github.com/rosinality/vq-vae-2-pytorch

In turn based on
https://github.com/deepmind/sonnet and ported it to PyTorch
"""

import torch
from torch import nn
from torch.nn import functional as F

def sq_l2(x, embedding):
    """
    Squared-L2 Distance. Return Tensor of shape [B*H*W, embeddings_dim

    Args:
        x (T)         : shape [B*H*W, C], where C = embeddings_dim
        embedding (T) : shape [embedding_dim, num_embeddings]
    """
    # Dist: squared-L2(p,q) = ||p||^2 + ||q||^2 - 2pq
    dist = (
        x.pow(2).sum(1, keepdim=True)
        - 2 * x @ embedding
        + embedding.pow(2).sum(0, keepdim=True)
    )
    _, embed_ind = (-dist).max(1)
    return embed_ind, dist


class Quantize(nn.Module):
    def __init__(self, dim, num_embeddings, decay=0.99, eps=1e-5, size=1):
        super().__init__()

        self.dim = dim
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.eps = eps
        self.size = size

        embed = torch.randn(self.size, self.size, dim, num_embeddings).normal_(0, 0.02)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', embed.clone())
        self.register_buffer('count', torch.zeros(num_embeddings).long())

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)

        # funky stuff in
        bs, hH, _, C = x.size()
        real_hH = hH // self.size

        x = x.view(bs, real_hH, self.size, real_hH, self.size, C).transpose(2,3).contiguous()

        # funky stuff out
        #x = x.transpose(3,2).reshape(bs, hH, hH, C)

        flatten = x.reshape(-1, self.size, self.size, self.dim)
        # Dist: squared-L2(p,q) = ||p||^2 + ||q||^2 - 2pq

        flatten = flatten.view(flatten.size(0), -1)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.view(-1, self.embed.size(-1))
            + self.embed.view(-1, self.embed.size(-1)).pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.num_embeddings)
        embed_onehot = embed_onehot.type(flatten.dtype) # cast
        embed_ind = embed_ind.view(*x.shape[:-3])
        quantize = self.embed_code(embed_ind)

        # calculate perplexity
        avg_probs  = F.one_hot(embed_ind, self.num_embeddings).view(-1, self.num_embeddings)
        avg_probs  = avg_probs.float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.training:
            decay = self.decay

            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(decay).add_(1 - decay, embed_sum.view(self.size, self.size, self.dim, self.num_embeddings))
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

            # bookkeeping
            self.count.data.add_(embed_onehot.sum(0).long())

        diff = (quantize.detach() - x).pow(2).mean()
        # The +- `x` is the "straight-through" gradient trick!
        quantize = x + (quantize - x).detach()

        # funky stuff out
        quantize = quantize.transpose(3,2).reshape(bs, hH, hH, C)

        quantize = quantize.permute(0, 3, 1, 2)
        #import pdb; pdb.set_trace()
        #test = self.idx_2_hid(embed_ind)

        return quantize, diff, embed_ind, perplexity


    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.permute(3, 0, 1, 2))


    def idx_2_hid(self, indices):
        out = self.embed_code(indices) # bs, H, W, s, s, C
        if self.size > 1:
            bs, H, C = out.size(0), self.size * out.size(1), out.size(-1)
            out = out.transpose(3, 2).reshape(bs, H, H, C)
        else:
            out = out.squeeze(-2).squeeze(-2)

        return out.permute(0, 3, 1, 2)


class GumbelQuantize(nn.Module):
    #def __init__(self, n_classes, decay_rate=0.9, decay_schedule=100, diff_temp=4.):
    def __init__(self, n_classes, decay_rate=0.9, decay_schedule=1000, diff_temp=4.):
        super().__init__()

        self.temp = diff_temp
        self.n_classes = n_classes
        self.min_temp  = 0.01
        self.decay_rate = decay_rate
        self.decay_schedule = decay_schedule
        self.batch_count = 0


    def temp_update(self):
        self.batch_count += 1
        if self.batch_count % self.decay_schedule == 0:
            self.temp=max(self.temp*self.decay_rate, self.min_temp)


    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)


    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size()).to(logits.device)
        return F.softmax(y / temperature, dim=-1)


    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            #return y.view(-1, latent_dim * categorical_dim)
            return y.view(y.size(0), -1)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        # return y_hard.view(-1, latent_dim * categorical_dim)
        return y_hard.view(y_hard.size(0), -1)


    def forward(self, x):
        """ We should probably to something similar as VQ
            i.e. use the last dimension as probs, and keep 2D structure """

        self.temp_update()
        temp = self.temp

        x   = x.permute(0, 2, 3, 1) # (bs, C, H, W) --> (bs, H, W, C)
        shp = x.shape
        x   = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))

        z_q = self.gumbel_softmax(x, temp, hard=True)
        z_q = z_q.view(shp)
        z_q = z_q.permute(0, 3, 1, 2)

        embed_ind = z_q.max(dim=1)[1]

        # calculate perplexity
        avg_probs  = F.one_hot(embed_ind, self.n_classes).view(-1, self.n_classes)
        avg_probs  = avg_probs.float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, 0., embed_ind, perplexity


    def idx_2_hid(self, indices):
        out = F.one_hot(indices, self.n_classes) # bs, H, W, C
        return out.permute(0, 3, 1, 2).float()


class AQuantize(nn.Module):
    """ Argmax autoencoder quantization step """
    def __init__(self, dim, decay=0.99, eps=1e-10):
        super().__init__()
        self.dim = self.num_embeddings = dim
        self.eps = eps


    def forward(self, x):
        # x is a (bs, C, H, W) tensor

        # we use the ReLU with divisive normalization
        x = F.relu(x)
        x = x / (x.sum(dim=1, keepdim=True) + self.eps)

        embed_ind = x.max(dim=1)[1]
        one_hot   = F.one_hot(embed_ind, num_classes=self.dim)
        one_hot   = one_hot.permute(0, 3, 1, 2) # (bs, H, W, C) --> (bs, C, H, W)
        one_hot   = one_hot.float()

        quantize  = x + (one_hot - x).detach()

        diff = (quantize.detach() - x).pow(2).mean()
        # The +- `x` is the "straight-through" gradient trick!
        quantize = x + (quantize - x).detach()

        # calculate perplexity
        avg_probs  = one_hot.mean(dim=(0, 2, 3))
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # calculate diversity penalty
        q_bar = x.mean(dim=(0, 2, 3))
        diversity = (q_bar * self.dim - 1.).pow(2).mean()

        return quantize, diversity, embed_ind, perplexity


    def idx_2_hid(self, indices):
        out = F.one_hot(indices, self.dim) # bs, H, W, C
        return out.permute(0, 3, 1, 2).float()


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x

        return out


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        in_channel = args.in_channel
        channel = args.channel
        stride = args.stride
        num_residual_layers = args.num_residual_layers
        num_residual_hiddens = args.num_residual_hiddens

        if stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        elif stride == 1:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1)
            ]

        for i in range(num_residual_layers):
            blocks += [ResBlock(channel, num_residual_hiddens)]

        blocks += [nn.ReLU(inplace=True)]

        # equivalent of `quantize_conv`
        blocks += [nn.Conv2d(channel, args.embed_dim, 1)]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_channel = args.embed_dim #args.in_channel
        out_channel = args.in_channel #args.out_channel
        channel = args.num_hiddens
        num_residual_hiddens = args.num_residual_hiddens
        num_residual_layers = args.num_residual_layers
        stride = args.stride

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(num_residual_layers):
            blocks += [ResBlock(channel, num_residual_hiddens)]

        blocks += [nn.ReLU(inplace=True)]

        if stride == 8:
            blocks += [
                nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ]

        if stride == 4:
            blocks += [
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ]

        elif stride == 2:
            blocks += [nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)]

        elif stride == 1:
            blocks += [nn.Conv2d(channel, out_channel, 3, padding=1)]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


