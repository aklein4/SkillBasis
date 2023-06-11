

class Config(dict):

    def __getattr__(self, attr):
        return self[attr]
    
    def inherit(self, other):
        for k in other.keys():
            if k not in self.keys():
                self[k] = other[k]


DefaultConfig = Config(

    state_dim = 5,
    action_dim = 2,

    n_skills = 2,

    hidden_dim = 32,
    n_layers = 2,

    dropout = 0.1

)


DefaultEncoder = Config(
)
DefaultEncoder.inherit(DefaultConfig)


DefaultDecoder = Config(
)
DefaultDecoder.inherit(DefaultConfig)


DefaultBasis = Config(
)
DefaultBasis.inherit(DefaultConfig)


DefaultBaseline = Config(
)
DefaultBaseline.inherit(DefaultConfig)


DefaultPolicy = Config(

    discrete = False

)
DefaultPolicy.inherit(DefaultConfig)


DefaultManager = Config(
)
DefaultManager.inherit(DefaultConfig)


DefaultManagerBaseline = Config(
)
DefaultManagerBaseline.inherit(DefaultConfig)