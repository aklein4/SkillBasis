
import torch

from replay_buffer import ReplayBuffer
from utils import DEVICE, np2torch, torch2np

import math


class SkillGenerator():
    def __init__(self, n_skills):
        """ Samples a vector randomly from an n-sphere """
        self.n_skills = n_skills

        # normal distribution
        center = torch.zeros(n_skills, device=DEVICE).float()
        self.dist = torch.distributions.Normal(
            center, center**0
        )

        # calculate the log surface area for log_prob
        self.log_s = math.log(2) + math.log(math.pi) * (self.n_skills / 2) - math.lgamma(self.n_skills / 2)


    def sample(self):
        # normalized on sphere
        v = self.dist.sample()
        return v / torch.norm(v, p=2)
    

    def log_prob(self, z):
        # surface area of the n-sphere
        return torch.full_like(z, self.log_s)


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
        skills = []

        # nograd for inference
        with torch.no_grad():
            for _ in range(n_episodes):

                # reset environment
                s = np2torch(self.env.reset()).float()
                seed = s

                z = skill
                if skill is None:
                    z = self.skill_generator.sample()

                while True:
                    self.env.render()

                    # z = self.skill_generator.sample()

                    pi = self.pi_model(s, z)
                    a = pi.sample()
                    # if not self.pi_model.config.discrete:
                    #     a = torch.clamp(a, self.pi_model.config.action_min, self.pi_model.config.action_max)

                    if greedy:
                        if self.pi_model.config.discrete:
                            a = torch.argmax(pi.probs)
                        else:
                            a = pi.loc
                    
                    og_prob = torch.exp(pi.log_prob(a))

                    # step environment
                    new_s, r, done, info = self.env.step(torch2np(a))
                    new_s = np2torch(new_s).float()

                    # another step or end
                    if done:
                        break

                    else:
                        seed_states.append(seed)
                        states.append(s)
                        next_states.append(new_s)
                        actions.append(a)
                        og_probs.append(og_prob)
                        skills.append(z)
                        
                        s = new_s

        return ReplayBuffer(
            seed_states,
            states,
            next_states,
            actions,
            og_probs,
            skills
        )
        
