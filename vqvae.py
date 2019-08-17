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
    def __init__(self, dim, num_embeddings, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, num_embeddings).normal_(0, 0.02)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', embed.clone())
        self.register_buffer('count', torch.zeros(num_embeddings).long())

    def forward(self, x):
        flatten = x.reshape(-1, self.dim)
        # Dist: squared-L2(p,q) = ||p||^2 + ||q||^2 - 2pq
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.num_embeddings)
        embed_onehot = embed_onehot.type(flatten.dtype) # cast
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
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

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class DiffQuantize(Quantize):
    def __init__(self, dim, num_embeddings, decay=0.99, eps=1e-5, diff_temp=4,
                decay_schedule=100, decay_rate=0.90, embed_grad_update=False):
        """
        Differentiable NearNeigh Quantization. As temperature decreases,
        converges to nearest neighbour

        Args:
            nn_temp: softmax temp w/ nearest neighbour logits
        """
        super().__init__(dim, num_embeddings, decay, eps)
        # self.temp = torch.tensor(diff_temp, requires_grad=False).to(device)
        # self.min_temp = torch.tensor(0.05, requires_grad=False).to(device)
        self.temp = diff_temp
        self.min_temp = 0.01
        self.batch_count = 0
        self.decay_schedule = decay_schedule
        self.decay_rate = decay_rate
        self.embed_grad_update=embed_grad_update

        # If updating Embed with gradient descent, register Embed as parameter
        if embed_grad_update:
            delattr(self, 'embed')
            embed = torch.randn(dim, num_embeddings)
            self.register_parameter('embed', torch.nn.Parameter(embed))

    def temp_update(self):
        self.batch_count += 1
        if self.batch_count % self.decay_schedule == 0:
            self.temp=max(self.temp*self.decay_rate, self.min_temp)


    def embed_ema_update(self, flatten, embed_onehot):
        self.cluster_size.data.mul_(self.decay).add_(
            1 - self.decay, embed_onehot.sum(0)
        )
        embed_sum = flatten.transpose(0, 1) @ embed_onehot
        self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
        self.embed.data.copy_(embed_normalized)

    def forward(self, x, temp=None):
        """
        Args:
            temp: will use this temp instead of self.temp, for debugging
        """
        # Maybe decay temp
        if temp is None:
            self.temp_update()
            temp = self.temp

        # Calc dist
        shape = x.shape
        flatten = x.reshape(-1, self.dim)

        embed_ind, distances = sq_l2(flatten, self.embed)
        embed_onehot = F.one_hot(embed_ind, self.num_embeddings)
        embed_onehot = embed_onehot.type(flatten.dtype) # cast

        # Shape discrete Tensor into `x` shape
        embed_ind = embed_ind.view(*x.shape[:-1])

        # Get weights to nearest neigh: Softmax with temp
        soft = F.softmax(-distances/temp, dim=1)

        #print(temp, soft.max(dim=1)[0].mean())
        # New Embedding: Softmax over neighbours
        soft_quantize = soft @ self.embed.t()
        soft_quantize = soft_quantize.view(x.shape)

        # hard : 
        quantize = self.embed_code(embed_ind)
        quantize = soft_quantize #+ (quantize - soft_quantize).detach()

        if self.embed_grad_update and False:
            diff = 0
        else:
            # Make x closer do new embeddings
            diff = (quantize.detach() - x).pow(2).mean()

        # Non-grad Embedding update
        if self.training and not self.embed_grad_update:
                self.embed_ema_update(flatten, embed_onehot)

        return quantize, diff, embed_ind


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
    def __init__(self, in_channel, channel, num_residual_layers, num_residual_hiddens, stride):
        super().__init__()
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

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, num_residual_layers, num_residual_hiddens, stride):
        super().__init__()

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


class VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()

        in_channel = args.input_size[0]
        channel    = args.num_hiddens
        num_residual_layers = args.num_residual_layers
        num_residual_hiddens = args.num_residual_hiddens
        embed_dim = args.embed_dim
        num_embeddings = args.num_embeddings
        decay = args.decay
        downsample = args.downsample
        num_codebooks = args.num_codebooks

        assert embed_dim % num_codebooks == 0, ("you need that last dimension"
                            " to be evenly divisible by the amt of codebooks")

        self.enc = Encoder(in_channel, channel, num_residual_layers,
                                    num_residual_hiddens, stride=downsample)

        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)
        self.dec = Decoder(embed_dim, in_channel, channel, num_residual_layers,
                                    num_residual_hiddens, stride=downsample)

        # build the codebooks
        self.quantize = nn.ModuleList([Quantize(embed_dim // num_codebooks, num_embeddings,
                                decay=decay) for _ in range(num_codebooks)])

        self.register_parameter('dec_log_stdv', torch.nn.Parameter(\
                                                        torch.Tensor([0.])))

    def forward(self, x):
        """
        Args:
            x (Tensor): shape BCHW
        """
        # `diff`: MSE(embeddings in z_e_s, closest in codebooks)
        # `z_q`, shape B*EMB_DIM*CHW, is neirest neigh embeddings to x
        z_q, diff, emb_idx, ppl = self.encode(x)

        # `dec`: decode `z_q` to `x` size, it is the image reconstruction
        dec = self.decode(z_q)

        return dec, diff, ppl

    def encode(self, x):
        # Encode x to continuous space
        pre_z_e = self.enc(x)
        # Project that space to the proper size for embedding comparison
        z_e = self.quantize_conv(pre_z_e)

        # Divide into multiple chunks to fit each codebook
        z_e_s = z_e.chunk(len(self.quantize), 1)

        z_q_s, argmins = [], []
        diffs, ppl = 0., 0.

        # `argmin`: the indices corresponding to closest embedding in codebook
        # `z_q`: same size as z_e_s but now holds the vectors from codebook
        # `diff`: MSE(embeddings in z_e_s, closest in codebooks)
        for z_e, quantize in zip(z_e_s, self.quantize):
            # z_e, change shape form  BCHW to BHWC
            z_q, diff, argmin = quantize(z_e.permute(0, 2, 3, 1))
            avg_probs = torch.mean(z_q, dim=0)
            perplexity = torch.exp(-torch.sum(\
                                    avg_probs * torch.log(avg_probs + 1e-10)))
            z_q_s   += [z_q]
            argmins += [argmin]
            diffs   += diff
            ppl     += perplexity

        # Stack the z_q_s and permute, now `z_q` has the same shape as the
        # first z_e
        z_q = torch.cat(z_q_s, dim=-1)
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, diffs, argmins, ppl

    def decode(self, quant):
        return self.dec(quant)

class DiffVQVAE(VQVAE):
    def __init__(self, args):
        super().__init__(args)

        embed_dim = args.embed_dim
        num_codebooks = args.num_codebooks
        num_embeddings = args.num_embeddings
        diff_temp=args.nn_temp
        decay_schedule=args.temp_decay_schedule
        decay_rate=args.temp_decay_rate
        embed_grad_update=args.embed_grad_update

        # build the codebooks
        self.quantize = nn.ModuleList([
                    DiffQuantize(
                        embed_dim // num_codebooks,
                        num_embeddings,
                        decay=args.decay,
                        diff_temp=diff_temp,
                        decay_schedule=decay_schedule,
                        decay_rate=decay_rate,
                        embed_grad_update=embed_grad_update)
                    for _ in range(num_codebooks)])

    @property
    def temp(self):
        return self.quantize[0].temp

class VQVAE_2(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        num_residual_layers=2,
        num_residual_hiddens=32,
        embed_dim=64,
        num_embeddings=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, num_residual_layers,
                                                num_residual_hiddens, stride=2)
        self.enc_t = Encoder(channel, channel, num_residual_layers,
                                                num_residual_hiddens, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, num_embeddings)
        self.dec_t = Decoder(embed_dim, embed_dim, channel,
                        num_residual_layers, num_residual_hiddens, stride=2)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, num_embeddings)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2,
                                                                    padding=1)
        self.dec = Decoder(embed_dim + embed_dim,in_channel,channel,
                num_residual_layers,num_residual_hiddens, stride=2)

    def forward(self, x):
        quant_t, quant_b, diff, _, _ = self.encode(x)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, x):
        enc_b = self.enc_b(x)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_b = self.quantize_b.embed_code(code_b)

        dec = self.decode(quant_t, quant_b)

        return dec
