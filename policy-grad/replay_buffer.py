
from typing import Any
import torch

from utils import DEVICE

import random


class ReplayBuffer:
    def __init__(self,
        states=None,
        next_states=None,
        actions=None,
        og_probs=None,
        genes=None,
        prev_genes=None,
        rewards=None,
        dones=None,
        d=None
    ):
        assert states is not None or d is not None, "Must provide states or d"

        self.d = {}
        if d is not None:
            self.d = d.copy()
            return

        # convert all to tensors
        self.d["states"] = torch.stack(states).to(DEVICE)
        self.d["next_states"] = torch.stack(next_states).to(DEVICE)

        self.d["actions"] = torch.tensor(actions).to(DEVICE)
        self.d["og_probs"] = torch.stack(og_probs).to(DEVICE)
        
        self.d["genes"] = torch.tensor(genes).to(DEVICE)
        self.d["prev_genes"] = torch.tensor(prev_genes).to(DEVICE)

        self.d["rewards"] = torch.tensor(rewards).to(DEVICE).float()
        self.d["dones"] = torch.tensor(dones, dtype=torch.bool).to(DEVICE)

        # store which elements to remove (disignated in loss function)
        self.d["importance"] = torch.zeros_like(self.dones).float().to(DEVICE)

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
    

    def reduce(self, new_size):

        # remove elements to reach new_size
        if new_size >= len(self):
            return

        keep = self.d["importance"].topk(new_size).indices

        for k in self.d.keys():
            self.d[k] = self.d[k][keep]


    def __add__(self, other):
        """ Concatenate two buffers into a new one

        Args:
            other (ReplayBuffer): Buffer to concatenate with

        Returns:
            ReplayBuffer: New replay buffer
        """

        # init new buffer
        out = ReplayBuffer(d=self.d)

        for k in out.d.keys():
            out.d[k] = torch.cat([out.d[k], other.d[k].clone()])

        return out
    

    def get_switch_perc(self):
        return torch.sum((self.prev_genes != self.genes).bool()).item() / len(self)
    
    def get_avg_skill_len(self):
        tot_skills = 1
        curr = self.genes[0]

        for i in range(len(self)):
            if self.genes[i] != curr:
                tot_skills += 1
                curr = self.genes[i]
        
        return len(self) / tot_skills