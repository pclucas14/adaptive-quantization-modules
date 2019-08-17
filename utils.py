import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from scipy.signal import savgol_filter
import torch
from torchvision.utils import make_grid

def show(img, path):
    img = make_grid(img.cpu().data)+0.5
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.savefig(path)
    plt.clf()

def plot_results(recon_error, ppl, path):
    fold = os.path.dirname(path)
    if not os.path.exists(fold):
            os.makedirs(fold)

    train_res_recon_error_smooth = savgol_filter(recon_error, 201, 7)
    train_res_perplexity_smooth = savgol_filter(ppl, 201, 7)

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1,2,1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')

    ax = fig.add_subplot(1,2,2)
    ax.plot(train_res_perplexity_smooth)
    ax.set_title('Smoothed Average codebook usage (perplexity).')
    ax.set_xlabel('iteration')

    # plt.savefig(path)
    fig.savefig(path)
    plt.clf()

def discretized_logistic(mean, logscale, binsize=1 / 256.0, sample=None):
    scale = torch.exp(logscale)
    sample = (torch.floor(sample / binsize) * binsize - mean) / scale
    logp = torch.log(torch.sigmoid(sample + binsize / scale) \
						- torch.sigmoid(sample) + 1e-7)
    return logp.sum(dim=(1,2,3))

def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

