
import torch

import gym

from replay_buffer import ReplayBuffer
from utils import DEVICE, np2torch, torch2np
from pacman import Pacman


R_SCALE = 1
OBS_SCALE = 1

MAX_LEN = 1000

class Environment():
    def __init__(self, epi_model, pi_model, discount=None):

        self.epi_model = epi_model
        self.pi_model = pi_model

        self.discount = discount

        # create environment
        self.env = Pacman()


    def sample(self, n_episodes):

        # set models modes
        self.epi_model.eval()
        self.pi_model.eval()

        # things to collect
        states = []
        actions = []
        genes = []
        og_probs = []
        returns = []
        total_returns = []

        # nograd for inference
        torch.no_grad()

        for _ in range(n_episodes):

            # reset environment
            s = np2torch(self.env.reset()[0]).reshape(-1).float() * OBS_SCALE

            rewards = []

            while True:

                # call models
                g = torch.argmax(self.epi_model(s)).item()

                pi = self.pi_model(s, g)
                a = pi.sample()
                og_prob = pi.probs[a].item()

                a = a.item()

                # step environment
                new_s, r, done, info, _ = self.env.step(a)
                new_s = np2torch(new_s).float().reshape(-1) * OBS_SCALE
                r *= R_SCALE

                # store transition                    
                states.append(s)
                actions.append(a)
                genes.append(g)
                og_probs.append(og_prob)
                rewards.append(r)

                # another step or end
                s = new_s
                if done or len(rewards) >= MAX_LEN:

                    total_returns.append(sum(rewards))
                    for i in range(2, len(rewards)+1):
                        rewards[-i] += rewards[-i+1] * self.discount
                    returns.extend(rewards)

                    break

        # grad for training
        torch.enable_grad()

        return ReplayBuffer(
            states,
            actions,
            genes,
            og_probs,
            returns
        ), total_returns
