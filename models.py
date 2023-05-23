
import torch
from torch import nn
from torch.nn import functional as F

import configs
from model_utils import SkipNet, MobileNet


class Option(nn.Module):
    def __init__(self, config=configs.DefaultOption):
        super().__init__()
        self.config = config

        self.net = SkipNet(
            config.state_dim,
            config.hidden_dim,
            config.option_dim,
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

        self.q_layers = nn.ModuleList([
            nn.Linear(config.option_dim, config.hidden_dim)
            for _ in range(config.num_layers)
        ])

        self.kv_layers = nn.ModuleList([
            nn.Linear(config.hidden_dim, 2*config.hidden_dim*config.num_pi)
            for _ in range(config.num_layers)
        ])

        self.f_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
            for _ in range(config.num_layers)
        ])

        self.att_layers = nn.ModuleList([
            nn.MultiheadAttention(
                config.hidden_dim,
                config.num_heads,
                dropout=0,
                batch_first=True,
                bias=False
            )
            for _ in range(config.num_layers)
        ])

        self.output_layer = nn.Linear(config.hidden_dim, config.action_dim)

        self.option_embeddings = nn.Embedding(config.num_options, config.option_dim)

    
    def _layer(self, x, o, i):
        batch_size = x.shape[0]

        q = self.q_layers[i](self.option_embeddings(o))
        kv = self.kv_layers[i](x).reshape(batch_size, self.config.num_pi, self.config.hidden_dim, 2)
        k, v = torch.chunk(kv, 2, dim=-1)

        y = self.att_layers[i](q, k, v)[0].reshape(batch_size, self.config.num_pi*self.config.hidden_dim)
        
        y = self.f_layers[i](y)

        return x + y
    

    def forward(self, s, o):

        x = self.input_layer(s)

        for i in range(self.config.num_layers):
            x = self._layer(x, o, i)

        l = self.output_layer(x)

        dist = torch.distributions.Categorical(logits=l)

        return dist