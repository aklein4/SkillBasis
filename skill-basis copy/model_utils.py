
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout):
        super().__init__()

        self.in_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Dropout(dropout),
            nn.ELU(),
        )
        
        self.mid_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(h_dim, h_dim),
                    nn.Dropout(dropout),
                    nn.ELU(),
                )
            for _ in range(n_layers-1)],

            nn.Sequential(
                    nn.Linear(h_dim, h_dim),
                    nn.ELU(),
            )
        )

        self.out_layer = nn.Sequential(
            nn.Linear(h_dim, out_dim)
        )


    def forward(self, x):
        h = self.in_layer(x)
        h = self.mid_layers(h)
        return self.out_layer(h)


class SkipNet(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout):
        super().__init__()

        self.in_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Dropout(dropout),
            nn.ELU(),
        )
        
        self.mid_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(2*h_dim, h_dim),
                    nn.Dropout(dropout),
                    nn.ELU(),
                )
            for _ in range(n_layers)]
        )

        self.out_layer = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.ELU(dropout),
            nn.Linear(h_dim, out_dim)
        )


    def forward(self, x):
        h = self.in_layer(x)
        prev = h
        for layer in self.mid_layers:
            temp = h
            h = layer(torch.cat([h, prev], dim=-1))
            prev = temp

        return self.out_layer(torch.cat([h, prev], dim=-1))
