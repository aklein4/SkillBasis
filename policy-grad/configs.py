

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultBaseline = Config(
    state_dim=8,

    hidden_dim=32,
    num_layers=2,
    dropout=0.1
)


DefaultEpiPolicy = Config(
    state_dim=8,
    num_g=1,

    hidden_dim=32,
    num_layers=2,
    dropout=0.1
)


DefaultPolicy = Config(
    state_dim=8,
    action_dim=4,

    num_g=1,
    rank_dim=2,

    hidden_dim=32,
    num_layers=2,
    dropout=0.1
)
