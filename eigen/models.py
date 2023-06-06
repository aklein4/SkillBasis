
import torch
from torch import nn
from torch.nn import functional as F

import configs
from model_utils import EigenNet, EigenTransformer, get2dEmbedding


class Baseline(nn.Module):
    def __init__(self, config=configs.DefaultBaseline):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.n_tokens, config.hidden_dim)

        self.positional_embedding = nn.Parameter(get2dEmbedding(config.state_size, config.hidden_dim).unsqueeze(0))
        self.positional_embedding.requires_grad = False

        self.cls = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

        self.transformer = EigenTransformer(
            config.hidden_dim,
            self.config.n_heads,
            self.config.rank,
            self.config.modes,
            self.config.n_layers,
            self.config.dropout
        )

        self.out_layer = nn.Linear(config.hidden_dim, 1)


    def forward(self, s, g):
        unbatch = False
        if s.dim() == self.config.state_dim:
            unbatch = True
            s = s.unsqueeze(0)

        # apply embeddings and flatten
        x = self.embeddings(s)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # add positional embedding and cls
        x = x + self.positional_embedding
        x = torch.cat([self.cls.repeat(x.shape[0], 1, 1), x], dim=1)

        # get output
        y = self.transformer(x, g)
        out = self.out_layer(y[:, 0])

        if unbatch:
            out = out.squeeze(0)

        return out
    

    def normalize(self):
        self.transformer.normalize()
    

class Policy(nn.Module):
    def __init__(self, config=configs.DefaultPolicy):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.n_tokens, config.hidden_dim)

        self.positional_embedding = nn.Parameter(get2dEmbedding(config.state_size, config.hidden_dim).unsqueeze(0))
        self.positional_embedding.requires_grad = False

        self.cls = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

        self.transformer = EigenTransformer(
            config.hidden_dim,
            self.config.n_heads,
            self.config.rank,
            self.config.modes,
            self.config.n_layers,
            self.config.dropout
        )

        self.out_layer = nn.Linear(config.hidden_dim, config.action_size)


    def forward(self, s, g):
        unbatch = False
        if s.dim() == self.config.state_dim:
            unbatch = True
            s = s.unsqueeze(0)

        # apply embeddings and flatten
        x = self.embeddings(s)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # add positional embedding and cls
        x = x + self.positional_embedding
        x = torch.cat([self.cls.repeat(x.shape[0], 1, 1), x], dim=1)

        # get output
        y = self.transformer(x, g)
        out = self.out_layer(y[:, 0])

        if unbatch:
            out = out.squeeze(0)

        dist = torch.distributions.Categorical(logits=out)

        return dist
    

    def normalize(self):
        self.transformer.normalize()