import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Buffer(nn.Module):
    def __init__(self, input_size, n_classes, max_idx=256., amt=0, dtype=torch.LongTensor):
        super().__init__()

        self.input_size = input_size
        self.n_classes  = n_classes
        self.dtype      = dtype

        bx = dtype(amt, *input_size).fill_(0)
        by = torch.LongTensor(amt).fill_(0)
        bt = torch.LongTensor(amt).fill_(0)
        bidx = torch.LongTensor(amt).fill_(0)
        bstep = torch.LongTensor(amt).fill_(0)

        self.n_samples = amt
        self.mem_per_sample = np.prod(input_size) * np.log2(max_idx) / np.log2(256.)
        self.n_memory  = amt * self.mem_per_sample

        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bt', bt)
        self.register_buffer('bidx', bidx)
        self.register_buffer('bstep', bstep)

        self.to_one_hot  = lambda x : x.new(x.size(0), n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    def expand(self, amt):
        """ used when loading a model from `pth` file and the amt of samples in the buffer don't align """
        self.__init__(self.input_size, self.n_classes, dtype=self.dtype, amt=amt)

    @property
    def x(self):
        return self.bx[:self.n_samples]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.n_samples])

    @property
    def t(self):
        return self.bt[:self.n_samples]

    def update(self, idx, value):
        self.bx[idx] = value

    def add(self, in_x, in_y, in_t, in_idx, in_step):
        """ concatenate a sample at the end of the buffer """

        in_x = in_x.detach()

        # convert int `in_t` to long tensor
        if type(in_t) == int:
            in_t = torch.LongTensor(in_x.size(0)).to(in_x.device).fill_(in_t)
        if type(in_step) == int:
            in_step = torch.LongTensor(in_x.size(0)).to(in_x.device).fill_(in_step)

        if self.bx.size(0) > in_x.size(0):
            swap_idx = torch.randperm(self.bx.size(0))[:in_x.size(0)]

            tmp_x, tmp_y, tmp_t, tmp_idx, tmp_step = self.bx[swap_idx], self.by[swap_idx], self.bt[swap_idx], self.bidx[swap_idx], self.bstep[swap_idx]

            # overwrite
            self.bx[swap_idx]    = in_x
            self.by[swap_idx]    = in_y
            self.bt[swap_idx]    = in_t
            self.bidx[swap_idx]  = in_idx
            self.bstep[swap_idx] = in_step

            in_x, in_y, in_t, in_idx, in_step = tmp_x, tmp_y, tmp_t, tmp_idx, tmp_step

        self.bx     = torch.cat((self.bx, in_x))
        self.by     = torch.cat((self.by, in_y))
        self.bt     = torch.cat((self.bt, in_t))
        self.bidx   = torch.cat((self.bidx, in_idx))
        self.bstep  = torch.cat((self.bstep, in_step))

        self.n_samples += in_x.size(0)
        self.n_memory  += in_x.size(0) * self.mem_per_sample


    def free(self, n_samples=None, idx=None):
        """ free buffer space. Assumes data is shuffled when added"""

        assert n_samples is not None or idx is not None, \
                'must specify amt of points to remove, or specific idx'

        if n_samples is None: n_samples = idx.size(0)

        if n_samples == 0:
            return

        n_samples = int(n_samples)
        import pdb
        assert n_samples <= self.n_samples, pdb.set_trace()

        if idx is not None:
            idx_to_keep = torch.ones_like(self.by)
            idx_to_keep[idx] = 0
            idx_to_keep = idx_to_keep.nonzero().squeeze(1)

            self.bx = self.bx[idx_to_keep]
            self.by = self.by[idx_to_keep]
            self.bt = self.bt[idx_to_keep]
            self.bidx = self.bidx[idx_to_keep]
            self.bstep = self.bstep[idx_to_keep]
        else:
            self.bx = self.bx[:-n_samples]
            self.by = self.by[:-n_samples]
            self.bt = self.bt[:-n_samples]
            self.bidx = self.bidx[:-n_samples]
            self.bstep = self.bstep[:-n_samples]

        self.n_samples -= n_samples
        self.n_memory  -= n_samples * self.mem_per_sample


    def sample(self, amt, exclude_task=None):
        if exclude_task is not None:
            raise NotImplementedError

        amt = int(amt)

        if amt == 0:
            self.sampled_indices = self.by[:0]
            return self.bx[:0], self.by[:0], self.bt[:0], self.bidx[:0], self.bstep[:0]

        if exclude_task is not None:
            valid_indices = (self.t != exclude_task).nonzero().squeeze()
            bx, by, bt, bidx = self.bx[valid_indices], self.by[valid_indices], self.bt[valid_indices], self.bidx[valid_indices]
        else:
            bx, by, bt, bidx, bstep = self.bx[:self.n_samples], self.by[:self.n_samples], self.bt[:self.n_samples], self.bidx[:self.n_samples], self.bstep[:self.n_samples]

        if bx.size(0) < amt:
            raise ValueError('trying to sample more points than available')
            return bx, by
        else:
            indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False)).to(bx.device)

            # TODO: Note make sure `exclude_task` flag is not used
            self.sampled_indices = indices
            return bx[indices], by[indices], bt[indices], bidx[indices], bstep[indices]

