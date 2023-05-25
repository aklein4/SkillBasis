

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultEpiPolicy = Config(
    state_dim=4,
    num_g=2,

    hidden_dim=16,
    num_layers=2,
    dropout=0.1
)


DefaultPolicy = Config(
    state_dim=4,
    action_dim=2,

    num_g=2,
    rank_dim=4,

    hidden_dim=16,
    num_layers=2,
    dropout=0.1
)
