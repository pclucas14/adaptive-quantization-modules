import sys
import os
import pdb
import numpy as np

accs = [[], []]
for path in os.listdir():
    if os.path.isdir(path):
        if 'NB1' in path:
            id = 0
        else:
            assert 'NB2' in path
            id = 1
        try:
            acc = np.loadtxt(os.path.join(path, 'test_acc.txt'))
            accs[id] += [acc.mean()]
        except:
            pass

for i in range(2):
    n_blocks = i + 1
    acc = np.array(accs[i])
    mean = acc.mean()
    stderr = acc.std() / np.sqrt(acc.shape[0])
    print(n_blocks, '{:.4f} += {:.4f}'.format(mean, stderr))
