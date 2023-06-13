
import torch
from torch import nn
from torch.nn import functional as F

import configs
import model_utils


class Encoder(nn.Module):
    def __init__(self, config=configs.DefaultEncoder):
        super().__init__()
        self.config = config

        self.net = model_utils.Net(
            config.obs_dim,
            config.hidden_dim,
            config.n_skills,
            config.n_layers,
            config.dropout
        )

    
    def forward(self, s):
        return self.net(s[...,:self.config.obs_dim]) * self.config.scale


class Policy(nn.Module):
    def __init__(self, config=configs.DefaultPolicy):
        super().__init__()
        self.config = config

        self.net = model_utils.Net(
            config.state_dim + config.n_skills,
            config.hidden_dim,
            config.action_dim,
            config.n_layers,
            config.dropout
        )

    
    def forward(self, s, z_val, z_attn):
        Q = self.Q(s, z_val, z_attn)
        
        return torch.distributions.Categorical(logits=Q / self.config.alpha)
    

    def Q(self, s, z_val, z_attn):
        inp = torch.cat([s, z_val * z_attn], dim=-1)
        return self.net(inp)
    

    def V(self, s, z_val, z_attn):
        Q = self.Q(s, z_val, z_attn)

        return self.config.alpha * torch.logsumexp(Q / self.config.alpha, dim=-1)
    

    def entropy(self, Q):

        dist = torch.distributions.Categorical(logits=Q / self.config.alpha)

        return dist.entropy()