
import torch

from replay_buffer import ReplayBuffer
import utils


class SkillGenerator():
    def __init__(self, n_skills):
        # for bernoulli distribution with p = 0.5
        self.probs = torch.ones([n_skills], device=utils.DEVICE) * 0.5
        # sample attentions from same shape
        self.attn = torch.zeros_like(self.probs)

    def sample(self, batch_size):
        # vals in {-1, 1}
        vals = (2 * torch.bernoulli(self.probs.unsqueeze(0).expand(batch_size, -1))) - 1
        # attns from L1 normed exponential distribution
        attn = self.attn.unsqueeze(0).expand(batch_size, -1).clone()
        attn.exponential_()
        # attn = torch.rand_like(attn)
        return vals, attn.detach() / attn.sum(dim=-1, keepdim=True)

    def log_prob(self, batch_size):
        return torch.log(torch.ones([batch_size], device=utils.DEVICE) * 0.5)


class Environment():
    def __init__(self, env, pi_model, skill_generator=None):
        self.pi_model = pi_model
        self.skill_generator = skill_generator
        self.env = env


    def sample(self, n_episodes, batch_size=1, skill_period=None, skill=None, greedy=False):
        assert (skill is not None) ^ (self.skill_generator is not None and skill_period is not None)

        # set models modes
        self.pi_model.eval()

        # things to collect every step
        states = []
        next_states = []
        actions = []
        z_vals = []
        z_attns = []

        # things to collect every skills
        seed_states = []
        end_states = []
        z_vals_traj = []
        z_attns_traj = []

        # nograd for inference
        with torch.no_grad():
            for _ in range(n_episodes):
                t = 1

                # reset environment
                s = utils.np2torch(self.env.reset(batch_size)).float()
                seed = s

                # get the starting skill
                z_val, z_attn = None, None
                if skill is None:
                    z_val, z_attn = self.skill_generator.sample(batch_size)
                else:
                    z_val, z_attn = skill

                while True:
                    self.env.render()

                    # sample action
                    pi = self.pi_model(s, z_val, z_attn)
                    a = pi.sample()
                    if greedy:
                        a = torch.argmax(pi.probs, dim=-1)

                    # step environment
                    new_s, r, done, info = self.env.step(utils.torch2np(a))
                    new_s = utils.np2torch(new_s).float()

                    # another end if done
                    if done:
                        break

                    # add transition to buffer
                    for i in range(batch_size):
                        states.append(s[i])
                        next_states.append(new_s[i])
                        actions.append(a[i])
                        z_vals.append(z_val[i])
                        z_attns.append(z_attn[i])

                    for i in range(batch_size):
                        seed_states.append(seed[i])
                        end_states.append(new_s[i])
                        z_vals_traj.append(z_val[i])
                        z_attns_traj.append(z_attn[i])

                    # if sampling skills, record and sample new
                    if skill_period is not None and t % skill_period == 0:
                        z_val, z_attn = self.skill_generator.sample(batch_size)
                        seed = new_s

                    t += 1
                    s = new_s

        # if not sampling skills, only return full buffer
        full_buffer = ReplayBuffer(
            states=states,
            next_states=next_states,
            actions=actions,
            z_vals=z_vals,
            z_attns=z_attns
        )
        if skill_period is None:
            return full_buffer

        # if sampling skills, return full buffer and trajectory buffer
        traj_buffer = ReplayBuffer(
            states=seed_states,
            next_states=end_states,
            z_vals=z_vals_traj,
            z_attns=z_attns_traj
        )
        return full_buffer, traj_buffer
        
