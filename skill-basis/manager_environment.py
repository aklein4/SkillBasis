
import torch

from manager_replay_buffer import ReplayBuffer
from utils import DEVICE, np2torch, torch2np


class Environment():
    def __init__(self, env, man_model, pi_model, update_period):

        self.man_model = man_model
        self.pi_model = pi_model

        self.update_period = update_period

        # create environment
        self.env = env


    def sample(self, n_episodes, greedy=False):

        # set models modes
        self.man_model.eval()
        self.pi_model.eval()

        # things to collect
        states = []
        next_states = []
        actions = []
        og_probs = []
        dones = []
        rewards = []
        returns = []

        # nograd for inference
        with torch.no_grad():
            for _ in range(n_episodes):

                t = 0
                curr_return = 0

                # reset environment
                seed = np2torch(self.env.reset()).float()
                s = seed

                z = self.man_model(s).sample()

                while True:
                    self.env.render()

                    pi = self.pi_model(s, z)
                    a = pi.sample()
                    if not self.pi_model.config.discrete:
                        a = torch.clamp(a, self.pi_model.config.action_min, self.pi_model.config.action_max)

                    if greedy:
                        if self.pi_model.config.discrete:
                            a = torch.argmax(pi.probs)
                        else:
                            a = pi.loc

                    # step environment
                    new_s, r, done, info = self.env.step(torch2np(a))
                    new_s = np2torch(new_s).float()

                    curr_return += r

                    # another step or end
                    t += 1
                    if done or t % self.update_period == 0:
                        
                        states.append(seed)
                        next_states.append(new_s)
                        actions.append(z)
                        og_probs.append(torch.exp(self.man_model(seed).log_prob(z)))
                        dones.append(done)
                        rewards.append(r)
                        
                        seed = new_s
                        z = self.man_model(seed).sample()

                    if done:
                        returns.append(curr_return)
                        break
                    
                    s = new_s

        return ReplayBuffer(
            states,
            next_states,
            actions,
            og_probs,
            dones,
            rewards,
        ), returns
        
