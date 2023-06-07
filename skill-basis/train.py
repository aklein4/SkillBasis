
import torch

from environment import Environment
from models import Policy, Baseline
from trainer import Trainer
from logger import Logger
from gridworld import GridWorld
import utils

# result directory
OUTPUT_DIR = 'test'
SAVE_EVERY = 16

# training parameters
NUM_ITERS = 2048 # number of training iterations

EPISODES_PER_ITER = 1 # episodes sampled per iteration
EPOCHS_PER_ITER = 1 # epochs trained per iteration

LR = 1e-4 # learning rate
BATCH_SIZE = 1 # batch size

DISCOUNT = 0.75 # discount factor
REWARD_SMOOTHING = 0.95 # reward logging momentum

MODES = [0]

GRID_SIZE = 4
NUM_FOOD_TYPES = 3
NUM_FOOD = 1
TIME_LIMIT = 128

def main():

    # fresh models
    baseline = Baseline()
    baseline = baseline.to(utils.DEVICE)

    pi_model = Policy()
    pi_model = pi_model.to(utils.DEVICE)
    # pi_model.load_state_dict(torch.load('test/pi_model.pt', map_location='cpu'))

    logger = Logger(pi_model, baseline, len(MODES), OUTPUT_DIR, SAVE_EVERY)

    grid = GridWorld(GRID_SIZE, NUM_FOOD_TYPES, NUM_FOOD, TIME_LIMIT)

    # fresh training utils
    env = Environment(pi_model, grid)
    trainer = Trainer(env, pi_model, baseline, logger)

    # train trial
    trainer.train(
        MODES,
        NUM_ITERS,
        EPISODES_PER_ITER,
        EPOCHS_PER_ITER,
        LR,
        BATCH_SIZE,
        DISCOUNT,
        REWARD_SMOOTHING
    )


if __name__ == '__main__':
    main()