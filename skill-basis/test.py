
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
OUT_DIR = "figs"

def main():

    # fresh models
    encoder_model = Encoder()
    encoder_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "encoder_model.pt"), map_location='cpu'))
    encoder_model = encoder_model.to(utils.DEVICE)

    basis_model = Basis()
    basis_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "basis_model.pt"), map_location='cpu'))
    basis_model = basis_model.to(utils.DEVICE)

    grid = torch.zeros(20, 20, 2)
    for i in range(-10, 10):
        for j in range(-10, 10):
            enc, _ = encoder_model(torch.tensor([i, j] + [0]*(encoder_model.config.state_dim-2)).float().to(utils.DEVICE))
            vals = (basis_model() @ enc.unsqueeze(-1)).squeeze(-1)

            grid[i+10, j+10] = vals[:2]

    grid -= torch.min(grid)
    grid /= torch.max(grid)

    grid = torch.cat([grid[:, :, :1], grid], dim=-1)

    plt.imshow(utils.torch2np(grid))
    plt.show()
    plt.clf()

    pi_model = Policy()
    pi_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "pi_model.pt"), map_location='cpu'))
    pi_model = pi_model.to(utils.DEVICE)

    rocket = Drone(discrete=False, render=True)
    env = Environment(rocket, pi_model)

    while True:

        skill = None
        while skill is None:
            skill = input("Enter skill: ")
            try:
                skill = [float(i.strip()) for i in skill.split(',')]
            except:
                continue
            skill = torch.tensor(skill).float().to(utils.DEVICE)
        
        batch = env.sample(1, skill=skill, greedy=False)

        l, _ = encoder_model(batch.states)
        l_next, _ = encoder_model(batch.next_states)
        delta_l = (l_next - l).detach()

        for i in range(2, len(batch)):
            delta_l[-i] += 0.9*delta_l[-i+1]


        L = basis_model(len(batch))
        L = L / torch.norm(L, p=2, dim=-1, keepdim=True)

        proj = torch.bmm(L, delta_l.unsqueeze(-1)).squeeze(-1)

        plt.plot(utils.torch2np(proj))
        plt.legend(["1", "2"])
        plt.show()
        plt.clf()


if __name__ == '__main__':
    main()