
import torch

from manager_environment import Environment
from models import Manager, ManagerBaseline, Policy
from manager_trainer import Trainer
from manager_logger import Logger
from drone import Drone
import utils

import numpy as np
import os


# result directory
INPUT_DIR = 'run'
OUTPUT_DIR = 'man'
SAVE_EVERY = 16

# training parameters
NUM_ITERS = 512 # number of training iterations
UPDATE_EVERY = 8

EPISODES_PER_ITER = 8 # episodes sampled per iteration
EPOCHS_PER_ITER = 1 # epochs trained per iteration

LR = 5e-4 # learning rate
BATCH_SIZE = 8 # batch size

SAMPLE_PERIOD = 8
DISCOUNT = 0.75

REWARD_SMOOTHING = 0.5 # reward logging momentum

BOXES = np.array([
    [-10, 20, -20, 0],
    [0, 10, 10, 20]
])


def main():

    # fresh models
    pi_model = Policy()
    pi_model.load_state_dict(torch.load(os.path.join(INPUT_DIR, 'pi_model.pt'), map_location='cpu'))
    pi_model = pi_model.to(utils.DEVICE)

    man_model = Manager()
    encoder_model = encoder_model.to(utils.DEVICE)
    # encoder_model.load_state_dict(torch.load('test/encoder_model.pt', map_location='cpu'))

    baseline_model = ManagerBaseline()
    baseline_model = baseline_model.to(utils.DEVICE)
    # baseline_model.load_state_dict(torch.load('test/baseline_model.pt', map_location='cpu'))

    logger = Logger(pi_model, man_model, baseline_model, OUTPUT_DIR, SAVE_EVERY)

    rocket = Drone(discrete=False, render=False, boxes=BOXES)
    env = Environment(rocket, man_model, pi_model, SAMPLE_PERIOD)

    trainer = Trainer(env, man_model, baseline_model, logger)

    # train trial
    trainer.train(
        NUM_ITERS,
        UPDATE_EVERY,
        EPISODES_PER_ITER,
        EPOCHS_PER_ITER,
        LR,
        BATCH_SIZE,
        DISCOUNT,
        REWARD_SMOOTHING
    )


if __name__ == '__main__':
    main()