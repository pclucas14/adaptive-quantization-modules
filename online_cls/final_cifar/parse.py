import sys
import os
import pdb
import numpy as np

for path in os.listdir():
    if os.path.isdir(path): #True:#try:
        """ extract relevant args """
        arg = str(open(os.path.join(path, 'args.json'), 'rb').read().splitlines()[0])

        recon_idx = arg.find('recon_th')
        recon_idx += arg[recon_idx:].find('[')
        recon_end = recon_idx + arg[recon_idx:].find(']')
        recon_th = float(arg[recon_idx + 1:recon_end])

        mem_idx = arg.find('mem_size') + len('mem_size') + 3
        mem_size = arg[mem_idx: mem_idx + 15].split(',')[:1][0]

        """ fetch results  """
        results = np.load(os.path.join(path, 'results.npy'))
        n_runs  = results.shape[0]

        # test set error
        results = results[:, -1]

        # fetch last test task accuracies
        final = results[:, -1]

        # average over tasks
        final = final.mean(axis=-1)

        # average over runs
        final_acc_mu, final_acc_std = final.mean(), final.std()

        # take the max over time, substract by last task
        forget = results.max(axis=1) - results[:, -1]

        # don't calculate last task in forgetting
        forget = forget[:, :-1]

        # average over tasks
        forget = forget.mean(axis=-1)

        forget_mu, forget_std = forget.mean(), forget.std()

        # divide by sqrt(n_runs) to get standard error
        final_acc_stderr = final_acc_std  / np.sqrt(n_runs)
        forget_stderr    = forget_std     / np.sqrt(n_runs)

        print('mem size : {}\t recon_th :{:.4f}\t acc {:.4f} += {:.4f}\t fgt {:.4f} += {:.4f}'.format(
            mem_size, recon_th, final_acc_mu, final_acc_stderr, forget_mu, forget_stderr))
    else:#except:
        pass
