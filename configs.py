

class Config(dict):
    """ A dictionary that can be accessed with dot notation """
    def __getattr__(self, attr):
        return self[attr]
    
""" Default network structure """
DefaultConfig = Config(

    # io dimensions
    n_inputs = 8,
    n_outputs = 1,

    # network size
    h_dim = 64,
    n_layers = 2,

    # training parameters
    dropout = 0.1

)


DefaultEncoderConfig = DefaultConfig.copy()
DefaultEncoderConfig["enc_dim"] = 16


DefaultDecoderConfig = Config(

    # io dimensions
    d_model = 16,
    nhead = 4,
    dim_feedforward = 64,
    dropout = 0.1,

    num_layers = 2,

    seq_len = 4,

    n_outputs = 4

)

