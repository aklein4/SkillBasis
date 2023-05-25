
import torch

from environment import Environment
from models import Policy, EpiPolicy
from trainer import Trainer
from logger import Logger
from utils import DEVICE

# result directory
OUTPUT_DIR = 'test'

# training parameters
NUM_ITERS = 150 # number of training iterations
EPSILON_DECAY_ITERS = 25 # iterations for epsilon 1 -> 0 linearly
ITERS_PER_TARGET = 4 # iterations between target network updates

EPISODES_PER_ITER = 1 # episodes sampled per iteration
EPOCHS_PER_ITER = 1 # epochs trained per iteration

VALS_PER_EPOCH = 1024 # samples trained on every epoch
BUFFER_SIZE = 1024 # maximium number of samples in buffer

LR = 1e-4 # learning rate
BATCH_SIZE = 8 # batch size

DISCOUNT = 0.75 # discount factor
SWITCH_PENALTY = 0.00 # penalty for switching options
ROLLING_BETA = 0.99 # parameter prob norm momentum
REWARD_SMOOTHING = 0.98 # reward logging momentum


def main():

    logger = Logger(OUTPUT_DIR)

    # fresh models
    q_model = EpiPolicy()
    q_model = q_model.to(DEVICE)
    pi_model = Policy()
    pi_model = pi_model.to(DEVICE)

    # fresh training utils
    env = Environment(q_model, pi_model)
    trainer = Trainer(env, q_model, pi_model, logger)

    # train trial
    trainer.train(
        NUM_ITERS,
        EPSILON_DECAY_ITERS,
        ITERS_PER_TARGET,
        EPISODES_PER_ITER,
        EPOCHS_PER_ITER,
        VALS_PER_EPOCH,
        BUFFER_SIZE,
        LR,
        BATCH_SIZE,
        DISCOUNT,
        SWITCH_PENALTY,
        ROLLING_BETA,
        REWARD_SMOOTHING
    )


if __name__ == '__main__':
    main()