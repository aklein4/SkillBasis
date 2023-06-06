
import torch

import gym

from replay_buffer import ReplayBuffer
from utils import DEVICE, np2torch, torch2np


class Environment():
    def __init__(self, pi_model, env, discount=None):

        self.pi_model = pi_model

        self.discount = discount

        # create environment
        self.env = env


    def sample(self, n_episodes, mode):

        # set models modes
        self.pi_model.eval()

        # things to collect
        states = []
        actions = []
        og_probs = []
        returns = []
        modes = []
        total_returns = []

        # nograd for inference
        torch.no_grad()

        for _ in range(n_episodes):

            # reset environment
            s = np2torch(self.env.reset([mode]))

            rewards = []

            while True:

                pi = self.pi_model(s, torch.tensor(mode).to(DEVICE))
                a = pi.sample().item()
                
                og_prob = pi.probs[a].item()

                # step environment
                new_s, r, done, info = self.env.step(a)
                new_s = np2torch(new_s)

                # store transition                    
                states.append(s)
                actions.append(a)
                og_probs.append(og_prob)
                rewards.append(r)
                modes.append(mode)

                # another step or end
                s = new_s
                if done:

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
            og_probs,
            returns,
            modes
        ), total_returns
