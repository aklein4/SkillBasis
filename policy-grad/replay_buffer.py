
from typing import Any
import torch

from utils import DEVICE

import random


class ReplayBuffer:
    def __init__(self,
        states=None,
        actions=None,
        genes=None,
        og_probs=None,
        returns=None,
        d=None
    ):
        assert states is not None or d is not None, "Must provide states or d"

        self.d = {}
        if d is not None:
            self.d = d.copy()
            return

        # convert all to tensors
        self.d["states"] = torch.stack(states).to(DEVICE)

        self.d["actions"] = torch.tensor(actions).to(DEVICE)
        self.d["genes"] = torch.tensor(genes).to(DEVICE)

        self.d["og_probs"] = torch.tensor(og_probs).to(DEVICE)

        self.d["returns"] = torch.tensor(returns).to(DEVICE).float()

        self.d["advantages"] = torch.zeros_like(self.returns)

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
    

    def get_switch_perc(self):
        return torch.sum((self.genes[1:] != self.genes[:-1]).bool()).item() / len(self)
    

    def get_avg_skill_len(self):
        tot_skills = 1
        curr = self.genes[0]

        for i in range(len(self)):
            if self.genes[i] != curr:
                tot_skills += 1
                curr = self.genes[i]
        
        return len(self) / tot_skills