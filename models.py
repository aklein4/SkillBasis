
import torch
import torch.nn as nn
import torch.nn.functional as F

import configs


class _ForwardModel(nn.Module):
    def __init__(self, config=configs.DefaultConfig):
        """ Base model for Q and Pi models

        Args:
            config (Config, optional): Network structure information. Defaults to DefaultConfig.
        """

        super().__init__()
        self.config = config

        # input dim to hidden dim
        self.input_layer = nn.Sequential(
            nn.Linear(self.config.n_inputs, self.config.h_dim // 2),
            nn.Dropout(self.config.dropout),
            nn.GELU()
        )

        # hidden layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.h_dim, self.config.h_dim // 2),
                nn.Dropout(self.config.dropout),
                nn.GELU()
            ) for _ in range(self.config.n_layers)
        ])

        # overloaded in child class
        self.output_layer = None


    def _network(self, x):
        """ A forward call through the network

        Args:
            x (tensor): State tensor (batched or not)

        Returns:
            tensor: Output of the network
        """

        # state -> hidden
        h = self.input_layer(x)
        prev = h

        # layers with skip connections
        for layer in self.layers:
            temp = h
            h = layer(torch.cat([h, prev], dim=-1))
            prev = temp

        # hidden -> output
        y = self.output_layer(torch.cat([h, prev], dim=-1))
        return y


    def forward(self, x):
        """ Overloadable forward call

        Args:
            x (tensor): State tensor (batched or not)

        Raises:
            NotImplementedError: Implemented in child class
        """
        raise NotImplementedError("Cannot call forward on abstract class _ForwardModel")


class BaselineModel(_ForwardModel):
    def __init__(self, config=configs.DefaultConfig):
        super().__init__(config)

        self.output_layer = nn.Linear(self.config.h_dim, 1)
    

    def forward(self, x):
        return self._network(x)
    

class ObsEncoder(_ForwardModel):
    def __init__(self, config=configs.DefaultEncoderConfig):
        super().__init__(config)

        self.output_layer = nn.Linear(self.config.h_dim, self.config.enc_dim)
    

    def forward(self, x):
        return self._network(x)


class PolicyModel(nn.Module):

    def __init__(self, encoder_config=configs.DefaultEncoderConfig, config=configs.DefaultDecoderConfig):
        super().__init__()

        self.config = config

        self.encoder = ObsEncoder(encoder_config)

        layer = nn.TransformerDecoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=self.config.num_layers,
        )

        embedding = torch.zeros(
            self.config.seq_len,
            self.config.d_model,
        ).unsqueeze(1)
        nn.init.xavier_uniform_(embedding)
        self.embedding = nn.Parameter(embedding)
    
        self.head = nn.Linear(self.config.d_model, self.config.n_outputs)

        mask = torch.zeros((self.config.seq_len, self.config.seq_len)).bool()
        for i in range(self.config.seq_len):
            mask[i, :i+1] = True
        self.temporal_mask = nn.Parameter(mask)
        self.temporal_mask.requires_grad = False


    def forward(self, obs, prev_actions):
        
        enc = self.encoder(obs)

        in_tokens = self.embedding.clone()
        in_tokens[1:1+prev_actions.shape[0], :, :self.config.n_outputs] = prev_actions.unsqueeze(1)

        pred = self.decoder(
            tgt = in_tokens,
            memory = enc,
            tgt_mask = self.temporal_mask,
        )

        logits = self.head(pred)

        dist = torch.distributions.Categorical(logits=logits)

        return dist



