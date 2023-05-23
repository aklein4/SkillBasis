

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultPolicy = Config(
    state_dim=4,
    action_dim=2,

    num_options=4,
    option_dim=8,
    num_pi=4,

    hidden_dim=128,
    num_layers=4,
    num_heads=4,

    dropout=0.1
)
