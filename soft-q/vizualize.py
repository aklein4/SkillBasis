
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import Environment
from models import Policy, Encoder
from drone import Drone
import utils

import os
import matplotlib.pyplot as plt
import random


LOAD_DIR = 'eight'

# RUNNERS = [
#     [1.0, 0.0],
#     [0.5, 0.5],
#     [0.0, 1.0],
#     [-0.5, 0.5],
#     [-1.0, 0.0],
#     [-0.5, -0.5],
#     [0.0, -1.0],
#     [0.5, -0.5]
# ]
RUNNERS = [
    [1.0, 0],
    [0.75, 0.25],
    [0.5, 0.5],
    [0.25, 0.75],
    [0, 1.0]
]
SCALE = 10


def main():

    pi_model = Policy()
    pi_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "pi_model.pt"), map_location='cpu'))
    pi_model = pi_model.to(utils.DEVICE)

    encoder_model = Encoder()
    encoder_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "encoder_model.pt"), map_location='cpu'))
    encoder_model = encoder_model.to(utils.DEVICE)

    rocket = Drone(discrete=True, render=False, max_t=3)
    env = Environment(rocket, pi_model)

    states = []

    # for runner in RUNNERS:
    #     run = runner + [0] * (encoder_model.config.n_skills - 2)

    #     loc = torch.tensor(run).float().to(utils.DEVICE).unsqueeze(0)

    #     batch = env.sample(1, skill=(loc, torch.ones_like(loc)), greedy=True)
    #     states.append(utils.torch2np(batch.states[:, :2]))

    for i in range(64):
        run = [0]*8
        run[0] = 1

        loc = torch.tensor(run).float().to(utils.DEVICE).unsqueeze(0)
        loc.exponential_()
        for i in range(8):
            loc[0, i] *= random.choice([1, -1])
        loc /= loc.sum()


        batch = env.sample(1, skill=(loc, torch.ones_like(loc)), greedy=False)
        states.append(utils.torch2np(batch.states[:, :2]))

    # for i in range(len(RUNNERS)):
    #     rg = (RUNNERS[i][0] + 1)/2
    #     b = (RUNNERS[i][1] + 1)/2
    #     c = (rg, rg, b)
    #     plt.plot(states[i][:, 0], states[i][:, 1], label=str(RUNNERS[i]))

    for i in range(32):
        # rg = (RUNNERS[i][0] + 1)/2
        # b = (RUNNERS[i][1] + 1)/2
        # c = (rg, rg, b)
        plt.plot(states[i][:, 0], states[i][:, 1], label=str(i))

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # plt.legend()
    plt.title("Diversity with 8 Skills")
    plt.show()


if __name__ == '__main__':
    main()