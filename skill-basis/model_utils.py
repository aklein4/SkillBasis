
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

import matplotlib.pyplot as plt


class EigenLinear(nn.Module):
    def __init__(self, dim, rank, modes):
        super().__init__()
        
        self.dim = dim
        self.rank = rank
        self.modes = modes

        self.vecs = nn.Parameter(torch.zeros((rank, dim)))
        nn.init.kaiming_uniform_(self.vecs, a=5**0.5)

        self.values = nn.Embedding(modes, rank)

        self.normalize()


    def forward(self, x, g=None):
        if g is None:
            x, g = x

        assert isinstance(g, torch.Tensor), "gene must be tensor"
        assert g.dim() in [0, 1], "gene must be single, batch, or interpole"

        # downcast into eigen space
        latent = F.linear(x, self.vecs)

        # get the correct eigen values
        values = None
        if g.dtype == torch.long or g.dtype == torch.int:
            # index along batch dimension
            values = self.values(g)
        else:
            # interpolate into singlet
            values = self.values(torch.arange(self.modes, device=g.device))
            values = torch.sum(values * g.unsqueeze(-1), dim=0)

        # get values to match latent dim
        if values.dim() == 1:
            values = values.view(*(1,)*(latent.dim()-1), values.shape[0])
        else:
            values = values.view(values.shape[0], *(1,)*(latent.dim()-2), values.shape[1])

        latent = latent * values

        # upcast into original space
        y = F.linear(latent, self.vecs.t())
        return y


    def normalize(self):

        # normalize vectors
        norm = torch.norm(self.vecs, dim=-1, p=2)
        self.vecs.data = self.vecs.data / norm.unsqueeze(-1)
        self.values.weight.data = self.values.weight.data * norm


class EigenAttention(nn.Module):
    def __init__(self, dim, n_heads, rank, modes):
        super().__init__()

        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.rank = rank
        self.modes = modes

        self.Q = EigenLinear(dim, rank, modes)
        self.K = EigenLinear(dim, rank, modes)
        self.V = EigenLinear(dim, rank, modes)

        self.out = EigenLinear(dim, rank, modes)


    def forward(self, x, g=None):
        if g is None:
            x, g = x

        # calcumalate attention components
        q = self.Q(x, g).reshape(*x.shape[:-1], self.n_heads, self.dim // self.n_heads)
        k = self.K(x, g).reshape(*x.shape[:-1], self.n_heads, self.dim // self.n_heads)
        v = self.V(x, g).reshape(*x.shape[:-1], self.n_heads, self.dim // self.n_heads)

        # attend
        hidden = None
        if utils.DEVICE == torch.device("cuda"):
            hidden = F._scaled_dot_product_attention(q, k, v)[0].reshape(x.shape)
        else:
            hidden = F.scaled_dot_product_attention(q, k, v).reshape(x.shape)

        # compute output
        y = self.out(hidden, g)

        return y
    

    def normalize(self):
        self.Q.normalize()
        self.K.normalize()
        self.V.normalize()
        self.out.normalize()
    

class EigenBlock(nn.Module):
    def __init__(self, dim, n_heads, rank, modes, dropout=0.0):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.rank = rank
        self.modes = modes
        self.dropout = dropout

        # attention module
        self.attn = EigenAttention(dim, n_heads, rank, modes)

        # feed forward module
        self.ff = nn.Sequential(
            EigenLinear(dim, rank, modes),
            nn.Dropout(dropout),
            nn.ELU(),
            EigenLinear(dim, rank, modes)
        )

        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.dropout_layer = nn.Dropout(dropout)


    def forward(self, x, g):

        # get attention
        attended = self.attn(x, g)

        # apply dropout, skip, and norm
        hidden = x + self.dropout_layer(attended)
        hidden = self.attn_norm(hidden)

        # get feed forward
        ffed = self.ff[0](hidden, g)
        ffed = self.ff[1](ffed)
        ffed = self.ff[2](ffed)
        ffed = self.ff[3](ffed, g)

        # apply dropout, skip, and norm
        y = hidden + self.dropout_layer(ffed)
        y = self.ff_norm(y)

        return y


    def normalize(self):
        self.attn.normalize()
        self.ff[0].normalize()
        self.ff[-1].normalize()


class EigenTransformer(nn.Module):
    def __init__(self, dim, n_heads, rank, modes, n_layers, dropout=0.0):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.rank = rank
        self.modes = modes
        self.n_layers = n_layers
        self.dropout = dropout

        self.layers = nn.Sequential(
            *[EigenBlock(dim, n_heads, rank, modes, dropout) for _ in range(n_layers)]
        )


    def forward(self, x, g):
        y = x
        for layer in self.layers:
            y = layer(y, g)
        return y


    def normalize(self):
        for layer in self.layers:
            layer.normalize()


class EigenNet(nn.Module):

    def __init__(self, in_dim, h_dim, out_dim, n_layers, rank, dropout):
        super().__init__()

        self.in_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        self.mid_layers = nn.Sequential(
            *[
                nn.Sequential(
                    EigenLinear(h_dim, rank),
                    nn.ELU(),
                    nn.Dropout(dropout)
                )
            for _ in range(n_layers)]
        )

        self.out_layer = nn.Sequential(
            nn.Linear(h_dim, out_dim)
        )


    def forward(self, x):
        h = self.in_layer(x)
        h = self.mid_layers(h)
        return self.out_layer(h)


    def normalize(self):
        for layer in self.mid_layers:
            layer[0].normalize()


    def getValues(self):
        return torch.stack([layer[0].values.data for layer in self.mid_layers], dim=0)


    def showValues(self, save=None):
        values = self.getValues()

        fig, ax = plt.subplots(values.shape[0], 1)

        for i in range(values.shape[0]):
            v = values[i].t()
            v /= torch.max(v, keepdim=True)[0]
            ax[i].imshow(utils.torch2np(v), vmin=0)

        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        
        plt.close(fig)


def get2dEmbedding(size, h):

    len = int(size ** (1/2))
    chunk = h // 2

    coef = 2 * torch.pi / len

    pos = torch.zeros((len, len, h))

    for i in range(len):
        for k in range(chunk):
            pos[i, :, k] = torch.sin(torch.tensor(coef * i * (k+1)))
    
    for i in range(len):
        for k in range(chunk):
            pos[:, i, k+chunk] = torch.sin(torch.tensor(coef * i * (k+1)))

    return pos.reshape(-1, h)


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
