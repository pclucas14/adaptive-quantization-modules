# VQVAE  code adapted from https://github.com/rosinality/vq-vae-2-pytorch
# Gumbel Softmax code from https://github.com/YongfeiYan/Gumbel_Softmax_VAE/

import math
import utils
import torch
from torch import nn
from torch.nn import functional as F


class Quantize(nn.Module):

    """
    Quantization operation for VQ-VAE. Also supports Tensor Quantization

    Args:
        dim (int)         : dimensionality of each latent vector (D in paper)
        num_embeddings    : number of embedding in codebook (K in paper)
        size (int tuple)  : height and dim of each quantized tensor.
                            Use (1,1) for standard vector quantization
        embed_grad_update : if True, codebook is not updated with EMA,
                            but with gradients as in the original VQVAE paper.
        decay             : \gamme in EMA updates for the codebook

    """
    def __init__(self, dim, num_embeddings, size=1, embed_grad_update=False,
                 decay=0.99, eps=1e-5) :
        super().__init__()

        if type(size) == int:
            size = (size, size)

        self.dim = dim
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.eps = eps
        self.size = size
        self.egu  = embed_grad_update

        embed = torch.randn(*self.size, dim, num_embeddings).uniform_(-.02, .02)

        if self.egu:
            self.register_parameter('embed', nn.Parameter(embed))
        else:
            self.register_buffer('embed', embed)
            self.register_buffer('cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('embed_avg', embed.clone())


    def forward(self, x):
        """
        Perform quantization op.

        Args:
            x (T)              : shape [B, C, H, W], where C = embeddings_dim
        Returns:
            quantize (T)       : shape [B, H, W, C], where C = embeddings_dim
            diff (float)       : commitment loss
            embed_ind          : codebook indices used in the quantization.
                                 this is what gets stored in the buffer
            perplexity (float) : codebook perplexity
        """

        # put channel axis before H and W
        x = x.permute(0, 2, 3, 1)

        # funky stuff in
        bs, hH, hW, C = x.size()
        real_hH = hH // self.size[0]
        real_hW = hW // self.size[1]

        x = x.view(bs, real_hH, self.size[0], real_hW, self.size[1], C)\
                .transpose(2,3).contiguous()

        flatten = x.reshape(-1, *self.size, self.dim)
        # Dist: squared-L2(p,q) = ||p||^2 + ||q||^2 - 2pq

        flatten = flatten.view(flatten.size(0), -1)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.view(-1, self.embed.size(-1)) +
            self.embed.view(-1, self.embed.size(-1)).pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.num_embeddings)
        embed_onehot = embed_onehot.type(flatten.dtype) # cast
        embed_ind = embed_ind.view(*x.shape[:-3])
        quantize = self.embed_code(embed_ind)

        # calculate perplexity
        avg_probs  = F.one_hot(embed_ind, self.num_embeddings)
        avg_probs  = avg_probs.view(-1, self.num_embeddings).float().mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        if self.training and not self.egu:
            # EMA codebook update
            decay = self.decay

            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(decay).add_(1 - decay, \
                    embed_sum.view(*self.size, self.dim, self.num_embeddings))
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) /\
                    (n + self.num_embeddings * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()

        if self.egu:
            # add vector quantization loss
            diff += (quantize - x.detach()).pow(2).mean()

        quantize = x + (quantize - x).detach()

        # reshape to match input shape
        quantize = quantize.transpose(3,2).reshape(bs, hH, -1, C)
        quantize = quantize.permute(0, 3, 1, 2)

        return quantize, diff, embed_ind, perplexity


    def embed_code(self, embed_id):
        """ fetch elements in the codebook """

        return self.embed.permute(3, 0, 1, 2)[embed_id]


    def idx_2_hid(self, indices):
        """ build `z_q` from the codebook indices """

        out = self.embed_code(indices)  # bs, H, W, s1, s2, C
        bs, hHs, hWs, _, _, C = out.shape
        if max(self.size) > 1:
            out = out.transpose(3, 2).reshape(bs, hHs * self.size[0], \
                    hWs * self.size[1], C)
        else:
            out = out.squeeze(-2).squeeze(-2)

        return out.permute(0, 3, 1, 2)



class GumbelQuantize(nn.Module):

    """
    Discretized the input using Gumbel Softmax

    Args:
        n_classes (int) : number of possible classes per latent vector
    """
    def __init__(self, n_classes):
        super().__init__()

        self.temp = 1. # as prescribed in Riemer & al.
        self.n_classes = n_classes
        self.min_temp  = 0.5
        self.batch_count = 0

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
        """
        Perform quantization op.

        Args:
            x (T)              : shape [B, C, H, W], where C = embeddings_dim
        Returns:
            quantize (T)       : shape [B, H, W, C], where C = embeddings_dim
            embed_ind          : codebook indices used in the quantization.
                                 this is what gets stored in the buffer
            perplexity (float) : codebook perplexity
        """
        temp = self.temp

        # put channel axis before H and W
        x   = x.permute(0, 2, 3, 1)
        shp = x.shape

        # broken down into bs, latent_dim, categorical dim
        x   = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))

        z_q = self.gumbel_softmax(x, temp)
        z_q = z_q.view(shp)
        z_q = z_q.permute(0, 3, 1, 2)

        embed_ind = z_q.max(dim=1)[1]

        # calculate perplexity
        avg_probs  = F.one_hot(embed_ind, self.n_classes)
        avg_probs  = avg_probs.view(-1, self.n_classes).float().mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        return z_q, 0., embed_ind, perplexity


    def idx_2_hid(self, indices):
        """ build `z_q` from the codebook indices """

        out = F.one_hot(indices, self.n_classes) # bs, H, W, C
        return out.permute(0, 3, 1, 2).float()


class AQuantize(nn.Module):
    """
    Quantization operator using a fixed / one-hot codebook, as in the paper
    "The challenge of realistic music generation: modelling raw audio at scale"

    Args:
        dim (int)         : dimensionality of each latent vector (D in paper)
    """
    def __init__(self, dim, eps=1e-10):
        super().__init__()
        self.dim = self.num_embeddings = dim
        self.eps = eps


    def forward(self, x):
        """
        Perform quantization op.

        Args:
            x (T)              : shape [B, C, H, W], where C = embeddings_dim
        Returns:
            quantize (T)       : shape [B, H, W, C], where C = embeddings_dim
            diversity          : diversity loss as in original paper
            embed_ind          : codebook indices used in the quantization.
                                 this is what gets stored in the buffer
            perplexity (float) : codebook perplexity
        """

        # we use the ReLU with divisive normalization
        x = F.relu(x)
        x = x / (x.sum(dim=1, keepdim=True) + self.eps)

        embed_ind = x.max(dim=1)[1]
        one_hot   = F.one_hot(embed_ind, num_classes=self.dim)

        # put channel axis before H, W
        one_hot   = one_hot.permute(0, 3, 1, 2).float()
        quantize  = x + (one_hot - x).detach()

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        # calculate perplexity
        avg_probs  = one_hot.mean(dim=(0, 2, 3))
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        # calculate diversity penalty
        q_bar = x.mean(dim=(0, 2, 3))
        diversity = (q_bar * self.dim - 1.).pow(2).mean()

        return quantize, diversity, embed_ind, perplexity


    def idx_2_hid(self, indices):
        out = F.one_hot(indices, self.dim) # bs, H, W, C
        return out.permute(0, 3, 1, 2).float()

