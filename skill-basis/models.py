
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
            config.state_dim,
            config.hidden_dim,
            config.latent_dim,
            config.n_layers,
            config.dropout
        )

    
    def forward(self, s):
        return self.net(s)
    

class Basis(nn.Module):
    def __init__(self, config=configs.DefaultBasis):
        super().__init__()
        self.config = config

        self.basis = nn.Parameter(torch.randn(config.n_skills, config.latent_dim))
        self.sigma = nn.Parameter(torch.ones(1))

    
    def forward(self, s):
        return (
            self.basis.unsqueeze(0).expand(s.shape[0], -1, -1),
            torch.exp(self.sigma).unsqueeze(0).expand(s.shape[0], self.config.n_skills)
        )
    

class Policy(nn.Module):
    def __init__(self, config=configs.DefaultPolicy):
        super().__init__()
        self.config = config

        self.net = model_utils.Net(
            config.state_dim + config.n_skills,
            config.hidden_dim,
            config.action_dim if config.discrete else config.action_dim * 2,
            config.n_layers,
            config.dropout
        )

    
    def forward(self, s, z):
        inp = torch.cat([s, z], dim=-1)

        out = self.net(inp)

        dist = None
        if self.config.discrete:
            dist = torch.distributions.Categorical(logits=out)
        
        else:
            mus, log_sigmas = torch.split(out, self.config.action_dim, dim=-1)
            dist = torch.distributions.Normal(mus, torch.exp(log_sigmas))

        return dist
    

class Baseline(nn.Module):
    def __init__(self, config=configs.DefaultBaseline):
        super().__init__()
        self.config = config

        self.net = model_utils.Net(
            config.state_dim + config.n_skills,
            config.hidden_dim,
            1,
            config.n_layers,
            config.dropout
        )

    
    def forward(self, s, z):
        inp = torch.cat([s, z], dim=-1)
        return self.net(inp)

