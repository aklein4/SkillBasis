
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import Environment
from models import Policy, Encoder, Basis
from drone import Drone
import utils

import os
import matplotlib.pyplot as plt


LOAD_DIR = 'test'

RUNNERS = [
    [1.0, 0.0],
    [0.5, 0.5],
    [0.0, 1.0],
    [-0.5, 0.5],
    [-1.0, 0.0],
    [-0.5, -0.5],
    [0.0, -1.0],
    [0.5, -0.5]
]


def main():

    pi_model = Policy()
    pi_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "pi_model.pt"), map_location='cpu'))
    pi_model = pi_model.to(utils.DEVICE)

    rocket = Drone(discrete=True, render=False)
    env = Environment(rocket, pi_model)

    states = []

    for runner in RUNNERS:
        curr_skill = torch.tensor(runner).to(utils.DEVICE)
        batch = env.sample(1, (curr_skill, torch.ones_like(curr_skill)), greedy=False)
        states.append(utils.torch2np(batch.states[:, :2]))

    for i in range(len(RUNNERS)):
        rg = (RUNNERS[i][0] + 1)/2
        b = (RUNNERS[i][1] + 1)/2
        c = (rg, rg, b)
        plt.plot(states[i][:, 0], states[i][:, 1], label=str(RUNNERS[i]))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()