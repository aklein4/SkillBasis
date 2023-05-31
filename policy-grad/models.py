
import torch
from torch import nn
from torch.nn import functional as F

import configs
from model_utils import Net, SkipNet

from enum import Enum


class Baseline(nn.Module):
    def __init__(self, config=configs.DefaultBaseline):
        super().__init__()
        self.config = config

        self.net = Net(
            config.state_dim,
            config.hidden_dim,
            1,
            config.num_layers,
            config.dropout
        )

    
    def forward(self, s):
        return self.net(s)


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
        dist = torch.distributions.Categorical(logits=self.net(s))
        return dist


class Policy(nn.Module):
    def __init__(self, config=configs.DefaultPolicy):
        super().__init__()
        self.config = config

        self.input_layer = nn.Linear(config.state_dim, config.hidden_dim)

        self.main_layers = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for l in range(config.num_layers)
        ])

        self.g_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.hidden_dim, config.rank_dim),
                    nn.Linear(config.rank_dim, config.hidden_dim)
                )
                for g in range(config.num_g)
            ])
            for l in range(config.num_layers)
        ])

        self.activation = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.ELU()
        )

        self.output_layer = nn.Linear(config.hidden_dim, config.action_dim)


    def _layer(self, x, g, i, disable_g=False):

        h = self.main_layers[i](x)
        if disable_g:
            return self.activation(h)

        if g is None:
            for c in range(self.config.num_g):
                h[:, c] += self.g_layers[i][c](x[:, c]) * 0.1
        else:
            h += self.g_layers[i][g](x) * 0.1

        return self.activation(h)
    

    def forward(self, s, g=None, disable_g=False):

        if g is None and not disable_g:
            s = s.unsqueeze(-2)
            s = s.expand(s.shape[0], self.config.num_g, s.shape[-1])

        if disable_g:
            self.input_layer.requires_grad_(True)
            self.main_layers.requires_grad_(True)
            self.output_layer.requires_grad_(True)
        else:
            self.input_layer.requires_grad_(False)
            self.main_layers.requires_grad_(False)
            self.output_layer.requires_grad_(False)

        x = self.input_layer(s)

        for i in range(self.config.num_layers):
            x = self._layer(x, g, i, disable_g)

        l = self.output_layer(x)

        dist = torch.distributions.Categorical(logits=l)

        return dist
    

    def get_kl(self, s):
        
        probs = self.forward(s).probs
        avg = torch.mean(probs, dim=-2, keepdim=True)

        kl = torch.sum(probs * torch.log(probs / avg), dim=-2)

        return torch.sum(torch.mean(kl, dim=0)).item()