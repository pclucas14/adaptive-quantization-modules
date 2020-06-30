# VQVAE  code adapted from https://github.com/rosinality/vq-vae-2-pytorch
# Gumbel Softmax code from https://github.com/YongfeiYan/Gumbel_Softmax_VAE/

import pdb
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

        self.i   = 0
        self.dim = dim
        self.eps = eps
        self.count = 1
        self.decay = decay
        self.egu  = embed_grad_update
        self.update_unused  = False
        self.num_codebooks  = num_codebooks
        self.num_embeddings = num_embeddings

        R = 1. / num_embeddings
        embed = torch.randn(num_codebooks, num_embeddings, dim).uniform_(-R, R)

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

        if indices.max() >= K: pdb.set_trace()

        encodings = F.one_hot(indices, K).float()
        quantized = torch.gather(self.embed, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        quantized = quantized.view_as(x)

        if self.training and not self.egu:
            self.i += 1

            # EMA codebook update
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=1)

            n = torch.sum(self.ema_count, dim=-1, keepdim=True)
            self.ema_count = (self.ema_count + self.eps) / (n + K * self.eps) * n

            dw = torch.bmm(encodings.transpose(1, 2), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embed = self.ema_weight / self.ema_count.unsqueeze(-1)

            if self.i > 10 and self.update_unused:
                unused = (self.ema_count < 1).nonzero()

                # reset unused vectors to random ones from the encoder batch
                unused_flat = unused[:, 0] * K + unused[:, 1]

                # get encodings
                enc_out = x_flat[unused[:, 0], torch.arange(unused.size(0))]

                ema_weight = self.ema_weight.view(-1, D)
                ema_weight[unused_flat] = enc_out

                self.ema_weight = ema_weight.view_as(self.ema_weight)
                self.ema_count[unused[:, 0], unused[:, 1]] = self.ema_count.mean()


        diff = (quantized.detach() - x).pow(2)# .mean()

        if self.egu:
            # add vector quantization loss
            diff += (quantized - x.detach()).pow(2).mean()

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

        quantized = quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W)
        diff      = diff.permute(1, 0, 4, 2, 3).reshape(B, C, H, W)

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


    def trim(self, n_embeds=None):
        # remove unused embeddings
        keep = self.ema_count > 0.1

        if n_embeds is None:
            n_embeds = 2 ** torch.log2(keep.sum(-1).max().float()).ceil().int().item()

        # keep last `n_embeds` most used
        N, K, D  = self.embed.size()
        keep_idx = self.ema_count.sort()[1][:, -n_embeds:]
        offset   = torch.arange(N).view(-1, 1).to(keep.device) * K
        flat_idx = (keep_idx + offset).view(-1)

        self.embed = self.embed.reshape(N * K, D)[flat_idx].reshape(N, n_embeds, D)
        self.ema_count = self.ema_count.reshape(N * K)[flat_idx].reshape(N, n_embeds)
        self.ema_weight = self.ema_weight.reshape(N * K, D)[flat_idx].reshape(N, n_embeds, D)

        return n_embeds


    def quantize(self, x):
        tr = self.training
        self.training = False
        z_q = self.forward(x)[0]
        self.training = tr
        return z_q


    def idx_2_hid(self, indices):
        """ build `z_q` from the codebook indices """

        out = self.embed_code(indices)
        return out


