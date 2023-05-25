
import torch
from torch import nn
from torch.nn import functional as F

import configs
from model_utils import Net, SkipNet

from enum import Enum


class EpiPolicy(nn.Module):
    def __init__(self, config=configs.DefaultEpiPolicy):
        super().__init__()
        self.config = config

        self.net = Net(
            config.state_dim,
            config.hidden_dim,
            config.num_g,
            config.num_layers,
            config.dropout
        )

    
    def forward(self, s):
        return self.net(s)


class Policy(nn.Module):
    def __init__(self, config=configs.DefaultPolicy):
        super().__init__()
        self.config = config

        self.input_layer = nn.Linear(config.state_dim, config.hidden_dim)

        self.main_layers = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(config.num_layers)
        ])

        self.g_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.hidden_dim, config.rank_dim),
                    nn.Linear(config.rank_dim, config.hidden_dim)
                )
                for _ in range(config.num_g)
            ])
            for _ in range(config.num_layers)
        ])

        self.activation = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.ELU()
        )

        self.output_layer = nn.Linear(config.hidden_dim, config.action_dim)


    def _layer(self, x, o, i):

        main = self.main_layers[i](x)

        g = None
        if isinstance(o, torch.Tensor) and o.numel() > 1:
            assert o.dim() == 1
            g = torch.stack([self.g_layers[i][o[c]](x[c]) for c in range(o.numel())])
        else:
            g = self.g_layers[i][o](x)

        return self.activation(main + g)
    

    def forward(self, s, o):

        x = self.input_layer(s)

        for i in range(self.config.num_layers):
            x = self._layer(x, o, i)

        l = self.output_layer(x)

        dist = torch.distributions.Categorical(logits=l)

        return dist
    

    def get_kl(self, s):
        outs = []

        for o in range(self.config.num_g):
            outs.append(self.forward(s, o).probs)
        
        outs = torch.stack(outs)
        avg = torch.mean(outs, dim=0, keepdim=True)

        kl = torch.sum(outs * torch.log(outs / avg), dim=-1)

        return torch.sum(torch.mean(kl, dim=0)).item()