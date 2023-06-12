
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


def main():

    # fresh models
    encoder_model = Encoder()
    encoder_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "encoder_model.pt"), map_location='cpu'))
    encoder_model = encoder_model.to(utils.DEVICE)
    encoder_model.eval()

    basis_model = Basis()
    basis_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "basis_model.pt"), map_location='cpu'))
    basis_model = basis_model.to(utils.DEVICE)
    basis_model.eval()

    pi_model = Policy()
    pi_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "pi_model.pt"), map_location='cpu'))
    pi_model = pi_model.to(utils.DEVICE)
    pi_model.eval()

    print(basis_model())

    for z in range(8):

        if input("Continue? ") == '':
            break

        grid = torch.zeros(20, 20)
        for i in range(-10, 10):
            for j in range(-10, 10):
                enc = encoder_model(torch.tensor([i, j] + [0]*(encoder_model.config.state_dim-2)).float().to(utils.DEVICE))
                vals = (basis_model() @ enc.unsqueeze(-1)).squeeze(-1)

                grid[i+10, j+10] = vals[z]

        grid -= torch.min(grid)
        grid /= torch.max(grid)

        plt.imshow(utils.torch2np(grid))
        plt.show()
        plt.clf()

    rocket = Drone(discrete=True, render=True)
    env = Environment(rocket, pi_model)

    while True:

        skill = None
        while skill is None:
            skill = input("Enter Location: ")
            try:
                loc = [float(i.strip()) for i in skill.split(',')]
                loc = torch.tensor(loc).float().to(utils.DEVICE).unsqueeze(0)
            except:
                continue
        
        skill = encoder_model(loc)
        skill /= torch.sum(torch.abs(skill))

        print(skill)

        attended = (skill, torch.ones_like(skill))
        batch = env.sample(1, 1, skill=attended, greedy=False)

        l = encoder_model(batch.states)
        l_next = encoder_model(batch.next_states)
        delta_l = l_next * 10

        L, _ = basis_model(len(batch))

        proj = torch.bmm(L, l_next.unsqueeze(-1)).squeeze(-1)

        log_probs = torch.abs(skill) * torch.log(2 * torch.sigmoid(torch.sign(skill) * proj))

        plt.plot(utils.torch2np(proj))
        plt.legend(["1", "2"])
        plt.show()
        plt.clf()


if __name__ == '__main__':
    main()