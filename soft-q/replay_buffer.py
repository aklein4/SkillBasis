
import torch

from utils import DEVICE

import random


class ReplayBuffer:
    def __init__(self, **kwargs):
        self.d = {}
        if 'd' in kwargs.keys():
            for k in kwargs['d'].keys():
                self.d[k] = kwargs['d'][k]

        for k in kwargs.keys():
            if k == 'd':
                continue
            self.d[k] = torch.stack(kwargs[k]).to(DEVICE)
            self.d[k].detach_()


    def __getattr__(self, k):
        if k == "d":
            return self.d
        return self.d[k]


    def __len__(self):
        """
        Number of elements in the buffer.
        """
        l = None
        for k in self.d.keys():
            if l is None:
                l = self.d[k].shape[0]
            else:
                assert l == self.d[k].shape[0], "All elements must have same length"
        
        return l
    

    def shuffle(self):
        """ Randomly permute the elements in the buffer.
        """

        # shuffle the indices
        shuffler = list(range(len(self)))
        random.shuffle(shuffler)
        
        # shuffle the elements
        for k in self.d.keys():
            self.d[k] = self.d[k][shuffler]


    def reduce(self, size):
        self.shuffle()
        for k in self.d.keys():
            self.d[k] = self.d[k][:size]


    def __getitem__(self, x):
        """ Get a batch of elements from the buffer.

        Args:
            x (tuple[int, int]): index and batch_size

        Returns:
            ReplayBuffer: Buffer with the batch of elements
        """
        index, batch_size = x

        # init batch
        out = ReplayBuffer(d=self.d)

        # fill batch
        for k in out.d.keys():
            out.d[k] = out.d[k][index:index+batch_size]

        return out
    

    def __add__(self, other):

        # init new buffer
        out = ReplayBuffer(d=self.d)

        # add other buffer
        for k in out.d.keys():
            out.d[k] = torch.cat([out.d[k], other.d[k]], dim=0)

        return out
    