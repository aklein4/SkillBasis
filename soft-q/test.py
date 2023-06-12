
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import Environment
from models import Policy, Encoder
from drone import Drone
from trainer import L_SCALE, CLAM
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

    pi_model = Policy()
    pi_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "pi_model.pt"), map_location='cpu'))
    pi_model = pi_model.to(utils.DEVICE)
    pi_model.eval()

    for z in range(encoder_model.config.n_skills):

        if input("Continue? ") == '':
            break

        
        grid = torch.zeros(20, 20)
        center = None
        for i in range(-10, 10):
            for j in range(-10, 10):
                enc = encoder_model(torch.tensor([i, j] + [0]*(encoder_model.config.state_dim-2)).float().to(utils.DEVICE))

                grid[i+10, j+10] = enc[z]
                if i == 0 and j == 0:
                    center = enc[z].item()

        # grid = torch.sigmoid((grid - center) * L_SCALE)

        plt.imshow(utils.torch2np(grid))
        plt.show()
        plt.clf()

    rocket = Drone(discrete=True, render=True, max_t=5)
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
        
        comb = encoder_model(loc)
        vals = torch.sign(comb)
        attn = torch.abs(comb) / torch.sum(torch.abs(comb))

        print("Skill: ", utils.torch2np(vals * attn))

        skill = (vals, attn)
        batch = env.sample(1, skill=skill, greedy=True)

        l_seed = encoder_model(batch.states[0])
        l = encoder_model(batch.states)
        delta_l = (l - l_seed.unsqueeze(0)) * L_SCALE

        logmoid = torch.log(
            torch.clamp(torch.sigmoid(vals * delta_l), min=CLAM)
        )
        log_probs = torch.sum(logmoid * attn, dim=-1)
        probs = torch.exp(log_probs)

        plt.plot(utils.torch2np(probs))
        plt.legend(["1", "2"])
        plt.show()
        plt.clf()


if __name__ == '__main__':
    main()