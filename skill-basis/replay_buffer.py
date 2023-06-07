
from typing import Any
import torch

from utils import DEVICE

import random


class ReplayBuffer:
    def __init__(self,
        seed_states=None,
        states=None,
        next_states=None,
        actions=None,
        skills=None,
        d=None
    ):
        assert states is not None or d is not None, "Must provide states or d"

        self.d = {}
        if d is not None:
            self.d = d.copy()
            return

        # convert all to tensors
        self.d["seed_states"] = torch.stack(seed_states).to(DEVICE)
        self.d["states"] = torch.stack(states).to(DEVICE)
        self.d["next_states"] = torch.stack(next_states).to(DEVICE)

        self.d["actions"] = torch.stack(actions).to(DEVICE)

        self.d["skills"] = torch.stack(skills).to(DEVICE)

        for k in self.d.keys():
            self.d[k].detach_()


    def __getattr__(self, k):
        if k == "d":
            return self.d
        return self.d[k]


    def __len__(self):
        """
        Number of elements in the buffer.
        """
        return self.d["states"].shape[0]
    

    def shuffle(self):
        """ Randomly permute the elements in the buffer.
        """

        # shuffle the indices
        shuffler = list(range(len(self)))
        random.shuffle(shuffler)
        
        # shuffle the elements
        for k in self.d.keys():
            self.d[k] = self.d[k][shuffler]


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
    