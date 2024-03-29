
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import Environment
from models import Policy, Encoder, Basis
from drone import Drone
import utils

import os
import matplotlib.pyplot as plt


LOAD_DIR = 'run'
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
            enc = encoder_model(torch.tensor([i, j]).float().to(utils.DEVICE)).unsqueeze(-1)
            vals = (basis_model() @ enc).squeeze()

            grid[i+10, j+10] = vals

    for i in [0, 1]:
        grid[:, :, i] -= torch.min(grid[:, :, i])
        grid[:, :, i] /= torch.max(grid[:, :, i])
    
    grid = torch.cat([grid[:, :, :1], grid], dim=-1)

    plt.imshow(utils.torch2np(grid))
    plt.savefig("vis.png")
    exit()

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
        
        env.sample(1, skill=skill, greedy=False)


if __name__ == '__main__':
    main()