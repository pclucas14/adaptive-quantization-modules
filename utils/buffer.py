import pdb
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Buffer(nn.Module):
    def __init__(self, input_size, n_classes, max_idx=256., amt=0, dtype=torch.LongTensor):
        super().__init__()

        self.input_size = input_size
        self.n_classes  = n_classes
        self.dtype      = dtype

        bx    = dtype(amt, *input_size).fill_(0)
        by    = torch.LongTensor(amt).fill_(0)
        bt    = torch.LongTensor(amt).fill_(0)
        bidx  = torch.LongTensor(amt).fill_(0)
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

    @torch.no_grad()
    def add(self, in_x, add_info, idx=None):

        """ concatenate a sample at the end of the buffer """

        in_x    = in_x.detach()
        in_y    = add_info['y']
        in_t    = add_info['t']
        in_idx  = add_info['bidx']
        in_step = add_info['step']

        # in_y, in_t, in_idx, in_step, idx=None):

        # convert int `in_t` to long tensor
        if type(in_t) == int:
            in_t = torch.LongTensor(in_x.size(0)).to(in_x.device).fill_(in_t)
        if type(in_step) == int:
            in_step = torch.LongTensor(in_x.size(0)).to(in_x.device).fill_(in_step)

        if idx is not None:
            assert 'BoolTensor' in idx.type(), pdb.set_trace()

            in_x    = in_x[idx]
            in_y    = in_y[idx]
            in_t    = in_t[idx]
            in_idx  = in_idx[idx]
            in_step = in_step[idx]

        if self.bx.size(0) > in_x.size(0):
            swap_idx = torch.randperm(self.bx.size(0))[:in_x.size(0)]

            tmp_x    = self.bx[swap_idx]
            tmp_y    = self.by[swap_idx]
            tmp_t    = self.bt[swap_idx]
            tmp_idx  = self.bidx[swap_idx]
            tmp_step = self.bstep[swap_idx]

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


    @torch.no_grad()
    def free(self, n_samples=None, idx=None):
        """ free buffer space. Assumes data is shuffled when added"""

        assert n_samples is not None or idx is not None, \
                'must specify amt of points to remove, or specific idx'

        if n_samples is None:
            n_samples = idx.size(0) if idx.ndim > 0 else 0

        if n_samples == 0:
            return 0, 0

        n_samples = int(n_samples)
        assert n_samples <= self.n_samples, pdb.set_trace()

        if idx is not None:
            class_removed = self.y[idx].sum(0)

            idx_to_keep = torch.ones_like(self.by)
            idx_to_keep[idx] = 0
            idx_to_keep = idx_to_keep.nonzero().squeeze(1)

            self.bx = self.bx[idx_to_keep]
            self.by = self.by[idx_to_keep]
            self.bt = self.bt[idx_to_keep]
            self.bidx = self.bidx[idx_to_keep]
            self.bstep = self.bstep[idx_to_keep]
        else:
            class_removed = self.y[-n_samples:].sum(0)

            self.bx = self.bx[:-n_samples]
            self.by = self.by[:-n_samples]
            self.bt = self.bt[:-n_samples]
            self.bidx = self.bidx[:-n_samples]
            self.bstep = self.bstep[:-n_samples]

        self.n_samples -= n_samples
        self.n_memory  -= n_samples * self.mem_per_sample

        return class_removed, n_samples * self.mem_per_sample


    def adjust_n_embeds(self, n_embeds):
        self.mem_per_sample = np.prod(self.input_size) * np.log2(n_embeds) / np.log2(256.)
        self.n_memory = self.n_samples * self.mem_per_sample


    @torch.no_grad()
    def try_and_remove(self, n_samples, class_counts):
        # figure out how much per class this means

        n_samples = min(n_samples, self.n_samples)

        if n_samples == 0:
            return 0, 0

        """ figure out how many samples per class should be removed """

        # sort classes w.r.t count
        class_count, class_id = torch.sort(class_counts, descending=True)

        gain = torch.zeros_like(class_count)
        gain[1:] = class_count[:-1] - class_count[1:]
        cum_gain = gain.cumsum(0)

        # get class counts for removal
        counts = torch.zeros(n_samples, self.n_classes).to(class_counts.device)

        # don't bother with classes having too few elems to reach n_samples
        valid_idx = cum_gain < n_samples
        counts[cum_gain[valid_idx], torch.arange(self.n_classes)[valid_idx]] = 1

        counts = counts.cumsum(0)
        cum_counts = counts.cumsum(0)
        total_cum_counts = cum_counts.sum(1)

        idx = (total_cum_counts < n_samples).sum()

        to_be_removed_counts = tbr_counts = cum_counts[(idx - 1).clamp_(min=0)]
        missing = int(n_samples - tbr_counts.sum())

        tbr_old = tbr_counts.clone()

        if missing != 0:
            # randomly assign the missing samples to available classes
            n_avail_classes = tbr_counts.nonzero().size(0)
            sample = torch.LongTensor(abs(missing)).random_(0, self.n_classes).to(counts.device)
            sample = sample % n_avail_classes
            tbr_counts[:n_avail_classes] += np.sign(missing) * sample.bincount(minlength=n_avail_classes)

        assert tbr_counts.sum() == n_samples, pdb.set_trace()

        """ remove class specific samples """

        # restore valid order
        tbr_counts = tbr_counts[class_id.sort()[1]]

        # buffer is already in random order, so just remove from the top
        class_total = self.y.cumsum(0)

        #       did we reach cap already?    get actual label
        tbr = ((class_total <= tbr_counts) & self.y.bool()).int() #.sum(0)

        tbr_idx = tbr.sum(1).nonzero().squeeze(-1)

        return self.free(idx=tbr_idx)


    @torch.no_grad()
    def sample(self, amt=None, y_samples=None):

        # one or the other
        if amt is None:

            assert y_samples is not None

            if y_samples.sum() == 0:
                return self.bx[:0], {'y': self.by[:0],
                                     't': self.bt[:0],
                                     'idx': self.bidx[:0],
                                     'bidx': self.bidx[:0],
                                     'step': self.bstep[:0]}

            # get the indices

            # simulate a shuffle (only needed on the y's. Should be fast)
            y = self.y.clone()
            shuffle_idx = torch.randperm(self.n_samples).to(y.device)
            reorder_idx = shuffle_idx.sort()[1]
            y = y[shuffle_idx]

            n_cls = y_samples.size(-1)
            class_total = y.cumsum(0)[:, :n_cls]

            #      did we reach cap already?    get actual label
            tbs = ((class_total <= y_samples) & y.bool()[:, :n_cls]).int()
            tbs_idx = tbs.sum(1).nonzero().squeeze(-1)

            # unshuffle
            tbs_idx = shuffle_idx[tbs_idx]
            # if not (self.by[tbs_idx].bincount(minlength=y_samples.size(0)) == y_samples).all(): pdb.set_trace()

            indices = tbs_idx
        else:
            raise NotImplementedError

            assert y_samples is None

            if amt == 0:
                return self.bx[:0], self.by[:0], self.bt[:0], self.bidx[:0], self.bstep[:0]

            indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False)).to(bx.device)

        return self.bx[indices], {'y': self.by[indices],
                                  't': self.bt[indices],
                                  'idx': indices,
                                  'bidx': self.bidx[indices],
                                  'step': self.bstep[indices]}


    @torch.no_grad()
    def sample_everything(self):
        BS = 32
        n_batches = self.n_samples // BS
        if self.n_samples != n_batches * BS: n_batches += 1

        for batch in range(n_batches):
            idx = range(batch * BS, min(self.n_samples, (batch+1) * BS))
            yield self.bx[idx], {'y': self.by[idx],
                                 't': self.bt[idx],
                                 'idx': idx,
                                 'bidx': self.bidx[idx],
                                 'step': self.bstep[idx]}


if __name__ == '__main__':
    import numpy as np

    for i in range(1000):
        print(i)

        B = np.random.randint(100)
        INPUT_SIZE = (3, 32, 32)
        N_CLASSES  = 10
        buf = Buffer(INPUT_SIZE, N_CLASSES, dtype=torch.FloatTensor)

        in_x = torch.FloatTensor(size=(B, ) + INPUT_SIZE).normal_()
        in_y = torch.FloatTensor(B).uniform_(0, N_CLASSES).long()
        in_t = in_idx = in_step = torch.zeros_like(in_y)

        buf.add(in_x, in_y, in_t, in_idx, in_step)

        class_counts = torch.FloatTensor(N_CLASSES).uniform_(0, 50).long()

        buf.try_and_remove(np.random.randint(100), class_counts)
