
import torch

from environment import Environment
from models import Policy, EpiPolicy, Baseline
from trainer import Trainer
from logger import Logger
from utils import DEVICE

# result directory
OUTPUT_DIR = 'test'

# training parameters
NUM_ITERS = 8*1024 # number of training iterations

EPISODES_PER_ITER = 1 # episodes sampled per iteration
EPOCHS_PER_ITER = 1 # epochs trained per iteration

LR = 1e-3 # learning rate
BATCH_SIZE = 8 # batch size

DISCOUNT = 0.8 # discount factor
REWARD_SMOOTHING = 0 # reward logging momentum


def main():

    logger = Logger(OUTPUT_DIR)

    # fresh models
    baseline = Baseline()
    baseline = baseline.to(DEVICE)
    epi_model = EpiPolicy()
    epi_model = epi_model.to(DEVICE)
    pi_model = Policy()
    pi_model = pi_model.to(DEVICE)

    # fresh training utils
    env = Environment(epi_model, pi_model)
    trainer = Trainer(env,baseline, epi_model, pi_model, logger)

    # train trial
    trainer.train(
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