
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Policy, Encoder
from drone import Drone
import utils

import os
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm


LOAD_DIR = 'eight'

# TARGETS = np.array(
#     [
#         [8, 6],
#         [-10, 3],
#         [-5, -5],
#         [0, 10],
#         [5, -7]
#     ],
#     dtype=np.float32
# )
TARGETS = np.array(
    [
        [4, 2],
        [4, 12],
        [-6, -6],
        [-7, -15],
        [0, -10]
    ],
    dtype=np.float32
)

IN_LOOP = True


def get_skill(enc, s, loc):
    comb = enc(loc) - enc(s)
    vals = torch.sign(comb)
    attn = torch.abs(comb) / torch.sum(torch.abs(comb))
    return vals, attn


def main():

    pi_model = Policy()
    pi_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "pi_model.pt"), map_location='cpu'))
    pi_model = pi_model.to(utils.DEVICE)
    pi_model.eval()

    encoder_model = Encoder()
    encoder_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "encoder_model.pt"), map_location='cpu'))
    encoder_model = encoder_model.to(utils.DEVICE)
    encoder_model.eval()

    env = Drone(discrete=True, render=False, max_t=20)

    s = utils.np2torch(env.reset(1)).float()

    for i in tqdm(range(TARGETS.shape[0])):
        targ = TARGETS[i] / 20
        loc = utils.np2torch(targ).unsqueeze(0)
        states = []

        vals, attn = get_skill(encoder_model, s, loc)

        while True:
            env.render()
            states.append(utils.torch2np(s[0, :2]))

            pi = pi_model(s, vals, attn)
            # a = pi.probs.argmax(dim=1, keepdim=False)
            a = pi.sample()

            s, _, done, _ = env.step(utils.torch2np(a))
            s = utils.np2torch(s).float()

            if IN_LOOP:
                vals, attn = get_skill(encoder_model, s, loc)

            if torch.norm(s[0, :2] - loc) < 0.01 or done:
                env.t = 0
                break

        states = np.stack(states)
        eh = plt.plot(states[:, 0], states[:, 1])
        plt.scatter(targ[0], targ[1], marker='o', s=50, facecolors='none', edgecolor=eh[0].get_color())

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # plt.legend()
    plt.title("Motion Planning to Multiple Waypoints")
    plt.show()


if __name__ == '__main__':
    main()