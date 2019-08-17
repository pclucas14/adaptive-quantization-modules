"""
Vector Quantization

References:
[1] Salimans, Tim, et al. "Pixelcnn++: Improving the pixelcnn with discretized
    logistic mixture likelihood and other modifications." ICLR 2017.
    https://arxiv.org/abs/1701.05517

[2] van den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation
    learning." Advances in Neural Information Processing Systems. 2017.
    https://arxiv.org/abs/1711.00937
"""
import argparse
import os
import math
import utils 
from progress import Progress
import torch
import torch.nn.functional as F
import torch.optim as optim
from vqvae import VQVAE, DiffVQVAE
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from data_ import *
from pydoc import locate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vq_vae_loss(args, x_prime, x, vq_loss, model):
    """
    Compute discretized recon loss, combined loss for training and bpd.

    Bits per dimension (bpd) is simply nats per pixel converted to base 2.
    -(NLL / num_pixels) / np.log(2.)
    """
    # Use Discretized Logistic as an alternative to MSE, see [1]
    log_pxz = utils.discretized_logistic(x_prime, model.dec_log_stdv,
                                                    sample=x).mean()

    # recon_error = torch.mean((data_recon - data)**2)/args.data_variance
    # loss = recon_error + vq_loss

    loss = -1 * (log_pxz / args.num_pixels) + args.commitment_cost * vq_loss
    elbo = - (args.KL - log_pxz) / args.num_pixels
    bpd  = elbo / np.log(2.)

    return loss, log_pxz, bpd

def evaluate(args, loss_func, pbar, valid_loader, model):
    """
    Evaluate validation set
    """
    model.eval()
    valid_bpd, valid_recon_error , valid_perplexity = [], [], []
    # Loop data in validation set
    for x, _ in valid_loader:

        x = x.to(args.device)

        x_prime, vq_loss, perplexity = model(x)

        loss, log_pxz, bpd = loss_func(args, x_prime, x, vq_loss, model)

        valid_bpd.append((-1)*bpd.item())
        valid_recon_error.append((-1)*log_pxz.item())
        valid_perplexity.append(perplexity.item())

    av_bpd = np.mean(valid_bpd)
    av_rec_err = np.mean(valid_recon_error)
    av_ppl = np.mean(valid_perplexity)
    pbar.print_eval(av_bpd)
    # pbar.print_train(av_rec_err=float(av_rec_err), av_ppl=float(av_ppl),
                                                        # increment=100)
    return av_bpd

def train_epoch(args, loss_func, pbar, train_loader, model, optimizer,
        train_bpd, train_recon_error , train_perplexity, EVAL_BATCH):
    """
    Train for one epoch
    """
    model.train()
    # Loop data in epoch
    for _ in range(10):
        for jj, (x, _) in enumerate(train_loader):
            for ii in range(10):
                # This break used for debugging
                if args.max_iterations is not None:
                    if args.global_it > args.max_iterations:
                        break

                x = x.to(args.device)

                # Get reconstruction and vector quantization loss
                # `x_prime`: reconstruction of `input`
                # `vq_loss`: MSE(encoded embeddings, nearest emb in codebooks)
                x_prime, vq_loss, perplexity = model(x)

                loss, log_pxz, bpd = loss_func(args, x_prime, x, vq_loss, model)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_bpd.append((-1)*bpd.item())
                train_recon_error.append((-1)*log_pxz.item())
                train_perplexity.append(perplexity.item())

                # Print Average every 100 steps
                if (args.global_it+1) % 100 == 0 and ii == 0:
                    x_prime = model(EVAL_BATCH[0])[0]
                    if (args.global_it+1) % 500 == 0 and ii == 0:
                        from PIL import Image
                        from torchvision.utils import save_image
                        rescale_inv = lambda x : x * 0.5 + 0.5
                        save_image(rescale_inv(torch.stack((x_prime[:32], EVAL_BATCH[0][:32]), 1).view(-1, 3, 32, 32)), 'tmp.png')
                        Image.open('tmp.png').show()
                    print('used : ', model.quantize[0].count.unique().shape, 'MSE : ', F.mse_loss(x_prime, EVAL_BATCH[0]).item())

                    av_bpd = np.mean(train_bpd[-100:])
                    av_rec_err = np.mean(train_recon_error[-100:])
                    av_ppl = np.mean(train_perplexity[-100:])

                    if args.model == 'vqvae':
                        pbar.print_train(bpd=float(av_bpd), rec_err=float(av_rec_err),
                                                                        increment=100)
                    elif args.model == 'diffvqvae':
                        pbar.print_train(bpd=float(av_bpd), temp=float(model.temp),
                                                                    increment=100)
            args.global_it += 1

def generate_samples(model, valid_loader):
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    (x, _) = next(iter(valid_loader))
    x = x.to(device)

    x_prime, vq_loss, perplexity = model(x)

    utils.show(x_prime, "results/valid_recon.png")
    utils.show(x, "results/valid_originals.png")

def main(args):
    ###############################
    # TRAIN PREP
    ###############################
    print("Loading data")
    #tr_loader, valid_loader, data_var, input_size = \
    #                            data.get_data(args.data_folder,args.batch_size)

    data__ = locate('data_.get_split_cifar10')(args)
    tr_loader, valid_loader, test_loader  = [CLDataLoader(elem, args, train=t) for elem, t in zip(data__, [True, False, False])]

    EVAL_BATCH = [y for y in [x for x in valid_loader][0]][0]
    EVAL_BATCH = [x.cuda() for x in EVAL_BATCH]

    for task, train_loader in enumerate(tr_loader):
        args.downsample = args.input_size[-1] // args.enc_height
        #args.data_variance = data_var
        print(f"Training set size {len(train_loader.dataset)}")
        #print(f"Validation set size {len(valid_loader.dataset)}")

        print("Loading model")
        if args.model == 'diffvqvae':
            model = DiffVQVAE(args).to(device)
        elif args.model == 'vqvae':
            model = VQVAE(args).to(device)
        print(f'The model has {utils.count_parameters(model):,} trainable parameters')

        optimizer = optim.Adam(model.parameters(),lr=args.learning_rate,
                                                                    amsgrad=False)

        print(f"Start training for {args.num_epochs} epochs")
        num_batches = math.ceil(len(train_loader.dataset)/train_loader.batch_size)
        pbar = Progress(num_batches, bar_length=10, custom_increment=True)

        # Needed for bpd
        args.KL = args.enc_height * args.enc_height * args.num_codebooks * \
                                                        np.log(args.num_embeddings)
        args.num_pixels  = np.prod(args.input_size)

        ###############################
        # MAIN TRAIN LOOP
        ###############################
        best_valid_loss = float('inf')
        train_bpd = []
        train_recon_error = []
        train_perplexity = []
        args.global_it = 0
        for epoch in range(args.num_epochs):
            pbar.epoch_start()
            train_epoch(args, vq_vae_loss, pbar, train_loader, model, optimizer,
                                    train_bpd, train_recon_error, train_perplexity, EVAL_BATCH)
            # loss, _ = test(valid_loader, model, args)
            # pbar.print_eval(loss)
            valid_loss = evaluate(args, vq_vae_loss, pbar, valid_loader, model)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_epoch = epoch
                torch.save(model.state_dict(), args.save_path)
            pbar.print_end_epoch()

        print("Plotting training results")
        utils.plot_results(train_recon_error, train_perplexity,
                                                            "results/train.png")

        print("Evaluate and plot validation set")
        generate_samples(model, valid_loader)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Vector Quantization')
    add = parser.add_argument

    # Data and training settings
    add('--data_folder', type=str, default="../cl-pytorch/data",
            help='Location of data (will download data if does not exist)')
    add('--dataset', type=str,
            help='Dataset name')
    add('--batch_size', type=int, default=10)
    add('--max_iterations', type=int, default=None,
            help="Max it per epoch, for debugging (default: None)")
    add('--num_epochs', type=int, default=40,
            help='number of epochs (default: 40)')
    add('--learning_rate', type=float, default=1e-3)

    # Quantization settings
    add('--num_codebooks', type=int, default=1,
            help='Number of codebooks')
    add('--embed_dim', type=int, default=64,
            help='Embedding size, `D` in paper')
    add('--num_embeddings', type=int, default=256,
            help='Number of embeddings to choose from, `K` in paper')
    add('--commitment_cost', type=float, default=0.25,
            help='Beta in the loss function')
    add('--decay', type=float, default=0.99,
            help='Moving av decay for codebook update')

    # VAVAE model, defaults like in paper
    add('--model', type=str, choices=['vqvae', 'diffvqvae'], default='vqvae')
    add('--enc_height', type=int, default=8,
            help="Encoder output size, used for downsampling and KL")
    add('--num_hiddens', type=int, default=128,
            help="Number of channels for Convolutions, not ResNet")
    add('--num_residual_hiddens', type=int, default = 32,
            help="Number of channels for ResNet")
    add('--num_residual_layers', type=int, default=2)

    # Diff NearNeigh settings
    add('--nn_temp', type=float, default=20.0, metavar='M',
            help='Starting diff. nearest neighbour temp (default: 1.0)')
    add('--temp_decay_rate', type=float, default=0.9, metavar='M',
            help='Nearest neighbour temp decay rate (default: 0.9)')
    add('--temp_decay_schedule', type=float, default=100, metavar='M',
            help='How many batches before decay (default: 100)')
    add('--embed_grad_update', action='store_true', default=False,
            help="If True, update Embed with gradient instead of EMA")

    # Misc
    add('--saved_model_name', type=str, default='vqvae_fixed.pt')
    add('--saved_model_dir', type=str, default='saved_models/')
    add('--seed', type=int, default=521)

    args = parser.parse_args()

    # Extra args
    args.device = device
    args.save_path = os.path.join(args.saved_model_dir, args.saved_model_name)
    utils.maybe_create_dir(args.saved_model_dir)

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)

