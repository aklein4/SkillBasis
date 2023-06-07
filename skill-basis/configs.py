

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultBaseline = Config(

    n_tokens = 5,
    state_size = 16,
    state_dim = 2,

    hidden_dim = 16,
    n_layers = 3,

    n_heads = 1,
    rank = 16,

    modes = 2,

    dropout = 0.1

)


DefaultPolicy = Config(

    n_tokens = 5,
    state_size = 16,
    state_dim = 2,

    action_size = 4,

    hidden_dim = 16,
    n_layers = 3,

    n_heads = 1,
    rank = 16,

    modes = 2,

    dropout = 0.1

)
