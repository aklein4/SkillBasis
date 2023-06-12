
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
            config.latent_dim,
            config.n_layers,
            config.dropout
        )

    
    def forward(self, s):
        return self.net(s[...,:self.config.obs_dim])


class Basis(nn.Module):
    def __init__(self, config=configs.DefaultBasis):
        super().__init__()
        self.config = config

        base = torch.zeros(config.n_skills, config.latent_dim)
        for i in range(self.config.n_skills):
            if i < config.latent_dim:
                base[i, i] = 1
            else:
                # rotationally symmetric sample
                v = torch.randn(config.latent_dim)
                base[i] = v / torch.norm(v, p=2)
        self.basis = nn.Parameter(base)


    def forward(self, batch_size=None):
        if batch_size is None:
            return self.basis / torch.norm(self.basis, p=2, dim=-1, keepdim=True)

        basis = self.basis.unsqueeze(0)
        basis = basis.expand(batch_size, -1, -1)

        norm = torch.norm(basis, p=2, dim=-1, keepdim=True)

        return basis / norm, torch.mean(norm)
    

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

    
    def forward(self, s, z_val, z_attn):

        inp = torch.cat([s, z_val * z_attn], dim=-1)
        out = self.net(inp)

        dist = None
        if self.config.discrete:
            dist = torch.distributions.Categorical(logits=out)
        
        else:
            mus = torch.tanh(out[..., :self.config.action_dim])
            log_sigmas = out[..., self.config.action_dim:]
            dist = torch.distributions.Normal(mus, torch.sigmoid(log_sigmas))
        
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

    
    def forward(self, s, z_val, z_attn):
        inp = torch.cat([s, z_val * z_attn], dim=-1)
        return self.net(inp)

