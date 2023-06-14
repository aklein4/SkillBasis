
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import Environment
from models import Policy, Encoder
from drone import Drone
from trainer import CLAM
import utils

import os
import matplotlib.pyplot as plt


LOAD_DIR = 'lfg_delta'


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

    grids = []

    for z in range(encoder_model.config.n_skills):

        # if input("Continue? ") == '':
        #     break

        grid = torch.zeros(20, 20)
        for i in range(-10, 10):
            for j in range(-10, 10):
                s = torch.tensor([i, j] + [0]*(encoder_model.config.state_dim-2)).float().to(utils.DEVICE)
                enc = encoder_model(s / 10)

                grid[i+10, j+10] = enc[z]

        grids.append(grid)


    fix, ax = plt.subplots(1, 2)
    for i in range(2):
        ax[i].imshow(utils.torch2np(grids[i]))
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    # plt.imshow(utils.torch2np(grid))
    plt.tight_layout()
    plt.suptitle("Latent Space with 2 Skills")
    plt.subplots_adjust(top=0.93)
    plt.show()
    plt.clf()

    rocket = Drone(discrete=True, render=True, max_t=2.5)
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
        
        comb = encoder_model(loc) - encoder_model(torch.zeros_like(loc))
        # comb = loc
        vals = torch.sign(comb)
        attn = torch.abs(comb) / torch.sum(torch.abs(comb))

        print("Skill: ", utils.torch2np(vals * attn))

        skill = (vals, attn)
        batch = env.sample(1, skill=skill, greedy=True)

        l = encoder_model(batch.states)
        l_next = encoder_model(batch.next_states)
        delta_l = l_next - l

        logmoid = torch.log(
            torch.clamp(2 * torch.sigmoid(vals * delta_l), min=CLAM)
        )
        log_probs = torch.sum(logmoid * attn, dim=-1)

        plt.plot(utils.torch2np(logmoid))
        plt.legend(["1", "2"])
        plt.show()
        plt.clf()


if __name__ == '__main__':
    main()