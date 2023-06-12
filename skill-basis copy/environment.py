
import torch

from replay_buffer import ReplayBuffer
import utils


class SkillGenerator():
    def __init__(self, n_skills):
        # values from uniform distribution in {-1, 1}
        self.probs = torch.ones([n_skills], device=utils.DEVICE) * 0.5
        self.up = torch.zeros_like(self.probs)
        self.down = -self.up
        # tensor to fill for skill attentions
        self.attn = torch.zeros_like(self.probs)

    def sample(self):
        vals = torch.where(torch.bernoulli(self.probs), self.up, self.down)
        self.attn.exponential_()
        return vals, self.attn.detach() / self.attn.sum()

    def log_prob(self, batch_size):
        return torch.log(torch.ones([batch_size], device=utils.DEVICE) * 0.5)


class Environment():
    def __init__(self, env, pi_model, skill_generator=None):

        self.pi_model = pi_model

        self.skill_generator = skill_generator

        # create environment
        self.env = env


    def sample(self, n_episodes, skill=None, greedy=False):
        assert skill is not None or self.skill_generator is not None

        # set models modes
        self.pi_model.eval()

        # things to collect
        seed_states = []
        states = []
        next_states = []
        actions = []
        og_probs = []
        z_vals = []
        z_attns = []

        # nograd for inference
        with torch.no_grad():
            for _ in range(n_episodes):

                # reset environment
                s = utils.np2torch(self.env.reset()).float()
                seed = s

                z_val, z_attn = skill
                if skill is None:
                    z_val, z_attn = self.skill_generator.sample()

                while True:
                    self.env.render()

                    pi = self.pi_model(s, z_val, z_attn)
                    a = pi.sample()

                    if greedy:
                        if self.pi_model.config.discrete:
                            a = torch.argmax(pi.probs)
                        else:
                            a = pi.loc
                    
                    og_prob = torch.exp(pi.log_prob(a))

                    # step environment
                    new_s, r, done, info = self.env.step(utils.torch2np(a))
                    new_s = utils.np2torch(new_s).float()

                    # another step or end
                    if done:
                        break

                    else:
                        seed_states.append(seed)
                        states.append(s)
                        next_states.append(new_s)
                        actions.append(a)
                        og_probs.append(og_prob)
                        z_vals.append(z_val)
                        z_attns.append(z_attn)
                        
                        s = new_s

        return ReplayBuffer(
            seed_states,
            states,
            next_states,
            actions,
            og_probs,
            z_vals,
            z_attns
        )
        
