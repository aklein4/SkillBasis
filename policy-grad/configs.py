

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultBaseline = Config(
    state_dim=225*2,

    hidden_dim=512,
    num_layers=2,
    dropout=0.1
)


DefaultEpiPolicy = Config(
    state_dim=225*2,
    num_g=2,

    hidden_dim=512,
    num_layers=2,
    dropout=0.1
)


DefaultPolicy = Config(
    state_dim=225*2,
    action_dim=4,

    num_g=2,
    rank_dim=8,

    hidden_dim=512,
    num_layers=2,
    dropout=0.1
)
