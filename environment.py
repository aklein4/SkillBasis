
import torch

import gym

from replay_buffer import ReplayBuffer
from utils import DEVICE, np2torch, torch2np
from models import POLICY_MODES


R_SCALE = 1/100


class Environment():
    def __init__(self, q_model, pi_model, switch_penalty=0):

        self.q_model = q_model
        self.pi_model = pi_model

        self.switch_penalty = switch_penalty

        # create environment
        self.env = gym.make("LunarLander-v2")


    def sample(self, n_episodes, epsilon=0.0):
        """ Sample n_episodes from the environment using greedy epsilon policy

        Args:
            n_episodes (int): number of episodes to sample
            epsilon (float, optional): Prob of random action. Defaults to greedy.

        Returns:
            ReplayBuffer, list: buffer of sampled data, list of episode returns
        """

        # set models modes
        self.q_model.eval()
        self.pi_model.eval()
        self.pi_model.set_mode(POLICY_MODES.NORMAL)

        # things to collect
        states = []
        next_states = []
        actions = []
        og_probs = []
        genes = []
        prev_genes = []
        rewards = []
        dones = []
        returns = []

        # nograd for inference
        torch.no_grad()

        for _ in range(n_episodes):

            # reset environment
            s = np2torch(self.env.reset()).float()
            prev_g = None

            # accumulate rewards
            curr_return = 0

            while True:

                # call models
                Q = self.q_model(s)
                if prev_g is None:
                    prev_g = torch.argmax(Q).item()
                Q[prev_g] -= self.switch_penalty
                
                g = torch.argmax(Q).item()
                if torch.rand(1) < epsilon:
                    g = torch.randint(Q.shape[0], (1,)).item()

                # select parameter
                pi = self.pi_model(s, g)
                a = pi.sample().item()

                # step environment
                new_s, r, done, info = self.env.step(a)
                new_s = np2torch(new_s).float()

                # handle reward
                r *= R_SCALE
                curr_return += r

                # store transition                    
                states.append(s)
                next_states.append(new_s)

                actions.append(a)
                og_probs.append(torch.exp(pi.log_prob(torch.tensor(a, device=DEVICE))))

                genes.append(g)
                prev_genes.append(prev_g)

                rewards.append(r)
                dones.append(bool(done))

                # another step or end
                s = new_s
                prev_g = g
                if done:
                    # only used for logging
                    returns.append(curr_return) 
                    break

        # grad for training
        torch.enable_grad()

        return ReplayBuffer(
            states,
            next_states,
            actions,
            og_probs,
            genes,
            prev_genes,
            rewards,
            dones
        ), returns
