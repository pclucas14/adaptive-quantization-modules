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
        avg_probs  = avg_probs.float().mean()
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

        return quantize, diff, embed_ind, perplexity

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.permute(3, 0, 1, 2))


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


