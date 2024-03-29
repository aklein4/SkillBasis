

class Config(dict):

    def __getattr__(self, attr):
        return self[attr]
    
    def inherit(self, other):
        for k in other.keys():
            if k not in self.keys():
                self[k] = other[k]


DefaultConfig = Config(

    state_dim = 6,
    action_dim = 2,

    n_skills = 2,

    hidden_dim = 32,
    n_layers = 2,

    dropout = 0.1

)


DefaultEncoder = Config(
)
DefaultEncoder.inherit(DefaultConfig)


DefaultBasis = Config(
)
DefaultBasis.inherit(DefaultConfig)


DefaultBaseline = Config(
)
DefaultBaseline.inherit(DefaultConfig)


DefaultPolicy = Config(

    discrete = False,

    action_min = -1.0,
    action_max = 1.0,

)
DefaultPolicy.inherit(DefaultConfig)


DefaultManager = Config(
)
DefaultManager.inherit(DefaultConfig)


DefaultManagerBaseline = Config(
)
DefaultManagerBaseline.inherit(DefaultConfig)