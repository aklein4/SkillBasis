
import torch

from replay_buffer import ReplayBuffer
import utils


class SkillGenerator():
    def __init__(self, n_skills):
        # for bernoulli distribution with p = 0.5
        self.probs = torch.ones([n_skills], device=utils.DEVICE) * 0.5
        # sample attentions from uniform in [0, 1]
        self.attn = torch.zeros_like(self.probs)

    def sample(self, batch_size):
        # vals in {-1, 1}
        vals = (2 * torch.bernoulli(self.probs.unsqueeze(0).expand(batch_size, -1))) - 1
        # attns samples from L1 normed exponential distribution
        attn = self.attn.unsqueeze(0).expand(batch_size, -1).clone()
        attn = torch.ones_like(attn) / attn.shape[-1]
        return vals, attn.detach() / attn.sum(dim=-1, keepdim=True)

    def log_prob(self, batch_size):
        return torch.log(torch.ones([batch_size], device=utils.DEVICE) * 0.5)


class Environment():
    def __init__(self, env, pi_model, skill_generator=None):

        self.pi_model = pi_model

        self.skill_generator = skill_generator

        # create environment
        self.env = env


    def sample(self, n_episodes, batch_size=1, skill=None, greedy=False):
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
                s = utils.np2torch(self.env.reset(batch_size)).float()
                seed = s

                z_val, z_attn = None, None
                if skill is None:
                    z_val, z_attn = self.skill_generator.sample(batch_size)
                else:
                    z_val, z_attn = skill

                while True:
                    self.env.render()

                    pi = self.pi_model(s, z_val, z_attn)
                    a = pi.sample()

                    if greedy:
                        if self.pi_model.config.discrete:
                            a = torch.argmax(pi.probs, dim=-1)
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
                        for i in range(batch_size):
                            seed_states.append(seed[i])
                            states.append(s[i])
                            next_states.append(new_s[i])
                            actions.append(a[i])
                            og_probs.append(og_prob[i])
                            z_vals.append(z_val[i])
                            z_attns.append(z_attn[i])
                        
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
        
