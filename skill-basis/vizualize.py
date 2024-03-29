
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import Environment
from models import Policy, Encoder, Basis
from drone import Drone
import utils

import os
import matplotlib.pyplot as plt


LOAD_DIR = 'run2'

RUNNERS = [
    [0.5, 0.0],
    [0.25, 0.25],
    [0.0, 0.5],
    [-0.25, 0.25],
    [-0.5, 0.0]
]

def main():

    pi_model = Policy()
    pi_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "pi_model.pt"), map_location='cpu'))
    pi_model = pi_model.to(utils.DEVICE)

    rocket = Drone(discrete=False, render=False)
    env = Environment(rocket, pi_model)

    states = []

    for runner in RUNNERS:
        batch = env.sample(1, torch.tensor(runner).to(utils.DEVICE), greedy=False)
        states.append(utils.torch2np(batch.states[:, :2]))

    for i in range(len(RUNNERS)):
        plt.plot(states[i][:, 0], states[i][:, 1], label=str(RUNNERS[i]))

    plt..xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.legend()
    plt.savefig("path.png")


if __name__ == '__main__':
    main()