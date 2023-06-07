
import torch

from environment import Environment
from models import Policy
from gridworld import GridWorld
import utils

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N_EPISODES = 100
N_POINTS = 10

GRID_SIZE = 4
NUM_FOOD_TYPES = 3
NUM_FOOD = 2
TIME_LIMIT = 64

FOOD_TARGET = [0, 1]

def main():

    pi_model = Policy()
    pi_model.load_state_dict(torch.load('test/pi_model.pt', map_location='cpu'))
    pi_model = pi_model.to(utils.DEVICE)

    grid = GridWorld(10, NUM_FOOD_TYPES, NUM_FOOD, TIME_LIMIT)
    env = Environment(pi_model, grid, 0)

    outcomes = []
    for i in tqdm(range(N_POINTS+1)):
        beta = i / N_POINTS

        dist = [beta, 1-beta]

        _, returns = env.sample(N_EPISODES, dist, FOOD_TARGET)

        outcomes.append(sum(returns)/N_EPISODES)

    plt.plot(np.linspace(0, 1, N_POINTS+1), outcomes)
    plt.show()

if __name__ == '__main__':
    main()