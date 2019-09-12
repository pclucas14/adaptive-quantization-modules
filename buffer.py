import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Buffer(nn.Module):
    def __init__(self, input_size, n_classes, dtype=torch.LongTensor):
        super().__init__()

        bx = dtype(0, *input_size).fill_(0)
        by = torch.LongTensor(0).fill_(0)
        bt = torch.LongTensor(0).fill_(0)

        self.n_samples = 0
        self.n_memory  = 0
        self.mem_per_sample = np.prod(input_size)

        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bt', bt)

        self.to_one_hot  = lambda x : x.new(x.size(0), n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[:self.n_samples]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.n_samples])

    @property
    def t(self):
        return self.bt[:self.n_samples]

    def add(self, in_x, in_y, in_t, swap_idx):
        """ concatenate a sample at the end of the buffer """

        # convert int `in_t` to long tensor
        in_t = torch.LongTensor(in_x.size(0)).to(in_x.device).fill_(in_t)

        if self.bx.size(0) > in_x.size(0):
            if swap_idx is None:
                swap_idx = torch.randperm(self.bx.size(0))[:in_x.size(0)]

            tmp_x, tmp_y, tmp_t = self.bx[swap_idx], self.by[swap_idx], self.bt[swap_idx]

            # overwrite
            self.bx[swap_idx]   = in_x
            self.by[swap_idx]   = in_y
            self.bt[swap_idx]   = in_t

            in_x, in_y, in_t = tmp_x, tmp_y, tmp_t

        self.bx = torch.cat((self.bx, in_x))
        self.by = torch.cat((self.by, in_y))
        self.bt = torch.cat((self.bt, in_t))

        self.n_samples += in_x.size(0)
        self.n_memory  += in_x.size(0) * self.mem_per_sample


    def free(self, n_samples):
        """ free buffer space. Assumes data is shuffled when added"""

        n_samples = int(n_samples)
        import pdb
        assert n_samples <= self.n_samples, pdb.set_trace()

        self.bx = self.bx[:-n_samples]
        self.by = self.by[:-n_samples]
        self.bt = self.bt[:-n_samples]

        self.n_samples -= n_samples
        self.n_memory  -= n_samples * self.mem_per_sample


    def sample(self, amt, exclude_task=None):

        amt = int(amt)
        if exclude_task is not None:
            valid_indices = (self.t != exclude_task).nonzero().squeeze()
            bx, by = self.bx[valid_indices], self.by[valid_indices]
        else:
            bx, by = self.bx[:self.n_samples], self.by[:self.n_samples]

        if bx.size(0) < amt:
            # return self.bx[:self.n_samples], self.by[:self.n_samples]
            return bx, by
        else:
            indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False)).to(bx.device)
            return bx[indices], by[indices]

