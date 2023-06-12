

class Config(dict):

    def __getattr__(self, attr):
        return self[attr]
    
    def inherit(self, other):
        for k in other.keys():
            if k not in self.keys():
                self[k] = other[k]


DefaultConfig = Config(

    # env params
    state_dim = 4,
    action_dim = 9,

    # skill params
    n_skills = 2,

    # model params
    hidden_dim = 32,
    n_layers = 2,
    dropout = 0.1

)


DefaultEncoder = Config(
    obs_dim = 2
)
DefaultEncoder.inherit(DefaultConfig)


DefaultPolicy = Config(
    alpha = 0.2
)
DefaultPolicy.inherit(DefaultConfig)

