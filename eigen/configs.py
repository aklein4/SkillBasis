

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultBaseline = Config(

    n_tokens = 6,
    state_size = 100,
    state_dim = 2,

    hidden_dim = 64,
    n_layers = 3,

    n_heads = 4,
    rank = 64,

    modes = 2,

    dropout = 0.1

)


DefaultPolicy = Config(

    n_tokens = 6,
    state_size = 100,
    state_dim = 2,

    action_size = 4,

    hidden_dim = 64,
    n_layers = 3,

    n_heads = 4,
    rank = 64,

    modes = 2,

    dropout = 0.1

)
