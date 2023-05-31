
import torch
import gym

from models import EpiPolicy, Policy
from utils import DEVICE, np2torch, torch2np

import numpy as np
import os
import matplotlib.pyplot as plt


LOAD_LOC = "./test"

R_SCALE = 1/100
OBS_SCALE = 1
MAX_LEN = 250


def test(epi_model, pi_model):
    env = gym.make("LunarLander-v2")

    # set models modes
    epi_model.eval()
    pi_model.eval()

    # things to collect
    probs = []

    # reset environment
    s = np2torch(env.reset()).float() * OBS_SCALE

    while True:

        # call models
        epi = epi_model(s)
        probs.append(torch2np(epi.probs))
        g = epi.sample().item()

        pi = pi_model(s, g)
        a = pi.sample().item()

        # step environment
        new_s, r, done, info = env.step(a)
        env.render()
        new_s = np2torch(new_s).float() * OBS_SCALE

        # another step or end
        s = new_s
        if done or len(probs) >= MAX_LEN:

            break

    return np.stack(probs)


def main():
    
    epi_model = EpiPolicy()
    epi_model = epi_model.to(DEVICE)
    epi_model.load_state_dict(torch.load(os.path.join(LOAD_LOC, "epi_model.pt")))
    
    pi_model = Policy()
    pi_model = pi_model.to(DEVICE)
    pi_model.load_state_dict(torch.load(os.path.join(LOAD_LOC, "pi_model.pt")))
    
    probs = test(epi_model, pi_model)
    
    plt.plot(probs)
    plt.title("Gene Probability Over Time")
    plt.ylabel("p(g|s)")
    plt.xlabel("t")
    plt.legend(["g_{}".format(i) for i in range(epi_model.config.num_g)])
    plt.savefig("test.png")
    
    
if __name__ == "__main__":
    main()