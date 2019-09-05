import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Buffer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args   = args
        buffer_size = args.mem_size
        print('buffer has %d slots' % buffer_size)

        #input_size = (args.layers[0].enc_height // args.layers[0].quant_size[0] ,) * 2

        #bx = torch.LongTensor(buffer_size, *input_size).to(args.device).fill_(0)
        bx = torch.FloatTensor(buffer_size, 2, 40, 512).to(args.device).fill_(0.)
        by = torch.LongTensor(buffer_size).to(args.device).fill_(0)
        bt = torch.LongTensor(buffer_size).to(args.device).fill_(0)

        self.current_index = 0
        self.n_seen_so_far = 0

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bt', bt)

        self.to_one_hot  = lambda x : x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])

    @property
    def t(self):
        return self.bt[:self.current_index]

    def display(self):
        from torchvision.utils import save_image
        from PIL import Image

        if 'cifar' in self.args.dataset:
            shp = (-1, 3, 32, 32)
        else:
            shp = (-1, 3, 128, 128)

        save_image((self.x.reshape(shp) * 0.5 + 0.5), 'tmp.png', nrow=int(self.current_index ** 0.5))
        Image.open('tmp.png').show()
        print(self.by[:self.current_index])

    def add_reservoir(self, x, y, t):
        n_elem = x.size(0)

        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bt[self.current_index: self.current_index + offset].fill_(t)

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        # perform overwrite op
        self.bx[idx_buffer] = x[idx_new_data]
        self.by[idx_buffer] = y[idx_new_data]
        self.bt[idx_buffer] = t

        self.n_seen_so_far += x.size(0)

    def sample(self, amt, exclude_task=None):
        if exclude_task is not None:
            valid_indices = (self.t != exclude_task).nonzero().squeeze()
            bx, by = self.bx[valid_indices], self.by[valid_indices]
        else:
            bx, by = self.bx[:self.current_index], self.by[:self.current_index]

        if bx.size(0) < amt:
            # return self.bx[:self.current_index], self.by[:self.current_index]
            return bx, by
        else:
            indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False)).to(self.args.device)
            return bx[indices], by[indices]

    def sample_from_task(self, amt, task):
        valid_indices = (self.t == task).nonzero().squeeze()
        bx, by = self.bx[valid_indices], self.by[valid_indices]
        indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False)).to(self.args.device)
        return bx[indices], by[indices]


    def split(self, amt):
        indices = torch.randperm(self.current_index).to(self.args.device)
        return indices[:amt], indices[amt:]


