# VQVAE  code adapted from https://github.com/rosinality/vq-vae-2-pytorch
# Gumbel Softmax code from https://github.com/YongfeiYan/Gumbel_Softmax_VAE/

import math
import utils
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, RelaxedOneHotCategorical, Normal


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
    def __init__(self, dim, num_embeddings, num_codebooks=1, size=1, embed_grad_update=False,
                 decay=0.99, eps=1e-5) :
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.count = 1
        self.decay = decay
        self.egu  = embed_grad_update
        self.num_codebooks  = num_codebooks
        self.num_embeddings = num_embeddings

        embed = torch.randn(num_codebooks, num_embeddings, dim).uniform_(-.02, .02)

        if self.egu:
            self.register_parameter('embed', nn.Parameter(embed))
        else:
            self.register_buffer('embed', embed)
            self.register_buffer('ema_count', torch.zeros(num_codebooks, num_embeddings))
            self.register_buffer('ema_weight', embed.clone())


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

        B, C, H, W = x.size()
        N, K, D = self.embed.size()

        import pdb
        assert C == N * D, pdb.set_trace()

        # B,N,D,H,W --> N, B, H, W, D
        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)

        # N, B, H, W, D --> N, BHW, D
        x_flat = x.detach().reshape(N, -1, D)

        distances = torch.baddbmm(torch.sum(self.embed ** 2, dim=2).unsqueeze(1) +
                          torch.sum(x_flat ** 2, dim=2, keepdim=True),
                          x_flat, self.embed.transpose(1, 2),
                          alpha=-2.0, beta=1.0)

        indices   = torch.argmin(distances, dim=-1)
        embed_ind = indices.view(N, B, H, W).transpose(1,0)

        encodings = F.one_hot(indices, K).float()
        quantized = torch.gather(self.embed, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        quantized = quantized.view_as(x)

        if self.training and not self.egu:
            # EMA codebook update

            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=1)

            n = torch.sum(self.ema_count, dim=-1, keepdim=True)
            self.ema_count = (self.ema_count + self.eps) / (n + K * self.eps) * n

            dw = torch.bmm(encodings.transpose(1, 2), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embed = self.ema_weight / self.ema_count.unsqueeze(-1)

        diff = (quantized.detach() - x).pow(2).mean()

        if self.egu:
            # add vector quantization loss
            diff += (quantized - x.detach()).pow(2).mean()

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

        quantized = quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W)

        # remove this after
        embed_ind = embed_ind

        return quantized, diff, embed_ind, perplexity


    def embed_code(self, embed_ind):
        """ fetch elements in the codebook """

        # do as in the code

        D = self.embed.size(-1)
        # B, N, H, W --> N, B, H, W
        B, N, H, W = embed_ind.size()
        embed_ind  = embed_ind.transpose(1,0)

        # N, B, H, W --> N, BHW
        flatten   = embed_ind.reshape(N, -1)
        quantized = torch.gather(self.embed, 1, flatten.unsqueeze(-1).expand(-1, -1, D))
        quantized = quantized.view(N, B, H, W, D)
        quantized = quantized.permute(1, 0, 4, 2, 3).reshape(B, N*D, H, W)

        return quantized

    def idx_2_hid(self, indices):
        """ build `z_q` from the codebook indices """

        out = self.embed_code(indices)
        return out


class SoftQuantize(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super(SoftQuantize, self).__init__()


        latent_dim = 1 # number of codebooks
        self.embedding = nn.Parameter(torch.Tensor(latent_dim, num_embeddings, embedding_dim))
        nn.init.uniform_(self.embedding, -1/num_embeddings, 1/num_embeddings)

    def forward(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.reshape(N, -1, D)

        distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                  torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                  x_flat, self.embedding.transpose(1, 2),
                                  alpha=-2.0, beta=1.0)
        distances = distances.view(N, B, H, W, M)

        dist = RelaxedOneHotCategorical(0.5, logits=-distances)
        #dist = RelaxedOneHotCategorical(1, logits=-distances)

        if self.training:
            samples = dist.rsample().view(N, -1, M)
        else:
            samples = torch.argmax(dist.probs, dim=-1)
            samples = F.one_hot(samples, M).float()
            samples = samples.view(N, -1, M)

        embed_ind = torch.argmax(dist.probs, dim=-1).squeeze(0)

        quantized = torch.bmm(samples, self.embedding)
        quantized = quantized.view_as(x)

        KL = dist.probs * (dist.logits + math.log(M))
        KL[(dist.probs == 0).expand_as(KL)] = 0

        # KL = KL.sum(dim=(0, 2, 3, 4)).mean()
        KL = KL.mean()

        avg_probs = torch.mean(samples, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

        # return quantize, diff, embed_ind, perplexity
        quantized = quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W)

        return quantized, KL, embed_ind, perplexity.sum()


    def embed_code(self, embed_id):
        """ fetch elements in the codebook """

        return self.embed.permute(3, 0, 1, 2)[embed_id]

    def idx_2_hid(self, indices):
        """ build `z_q` from the codebook indices """

        emb = self.embedding.squeeze()

        max_ = int(indices.max())
        size = [int(x) for x in emb.size()]
        out = emb[indices]
        '''
        if indices.max() > emb.size(-1):
            import pdb; pdb.set_trace()
            xx = 1
        try:
            out = emb[indices]
        except:
            import pdb; pdb.set_trace()
            xx = 1
        '''

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


class CQuantize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #x = x.permute(0, 2, 3, 1)
        return x, 0, x, 0.

    def idx_2_hid(self, indices):
        return indices
