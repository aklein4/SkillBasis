
import torch

from environment import Environment
from models import Policy
from drone import Drone
import utils

import os


LOAD_DIR = 'test'


def main():

    # fresh models
    pi_model = Policy()
    pi_model.load_state_dict(torch.load(os.path.join(LOAD_DIR, "pi_model.pt"), map_location='cpu'))
    pi_model = pi_model.to(utils.DEVICE)

    rocket = Drone(discrete=True, render=True)
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
        
        env.sample(1, skill=skill)


if __name__ == '__main__':
    main()