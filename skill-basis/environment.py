
import torch

from replay_buffer import ReplayBuffer
from utils import DEVICE, np2torch, torch2np


class SkillGenerator():
    def __init__(self, n_skills, sigma):
        self.dist = torch.distributions.Normal(
            torch.zeros(n_skills, device=DEVICE),
            sigma * torch.ones(n_skills, device=DEVICE)
        )

    def sample(self):
        return self.dist.sample()
    
    def log_prob(self, z):
        return self.dist.log_prob(z)


class Environment():
    def __init__(self, env, pi_model, skill_generator=None):

        self.pi_model = pi_model

        self.skill_generator = skill_generator

        # create environment
        self.env = env


    def sample(self, n_episodes, skill=None):
        if skill is None and self.skill_generator is None:
            raise ValueError("Skill or generator must be provided.")

        # set models modes
        self.pi_model.eval()

        # things to collect
        seed_states = []
        states = []
        next_states = []
        actions = []
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
                    if not self.pi_model.config.discrete:
                        a = torch.clamp(a, self.pi_model.config.action_min, self.pi_model.config.action_max)

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
                        skills.append(z)

        return ReplayBuffer(
            seed_states,
            states,
            next_states,
            actions,
            skills
        )
        
