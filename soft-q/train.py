
import torch

from environment import Environment, SkillGenerator
from models import Policy, Encoder, Basis, Baseline
from trainer import Trainer
from logger import Logger
from drone import Drone
import utils


# result directory
OUTPUT_DIR = 'test'
SAVE_EVERY = 16

# training parameters
N_ITERS = 2048*2
UPDATE_EVERY = 8

N_EPISODES = 2
SAMPLE_BATCH_SIZE = 4
SKILL_EPOCHS = 1
PI_EPOCHS = 1

LR = 1e-4
BATCH_SIZE = 16

DISCOUNT = 0.75
ALPHA_ENTROPY = 0.1

SMOOTHING = 0.95


def main():

    # fresh models
    pi_model = Policy().to(utils.DEVICE)
    encoder_model = Encoder().to(utils.DEVICE)
    basis_model = Basis().to(utils.DEVICE)
    baseline_model = Baseline().to(utils.DEVICE)

    logger = Logger(pi_model, encoder_model, basis_model, baseline_model, OUTPUT_DIR, SAVE_EVERY)

    rocket = Drone(discrete=True, render=False)
    skill_gen = SkillGenerator(pi_model.config.n_skills)
    env = Environment(rocket, pi_model, skill_gen)

    trainer = Trainer(env, pi_model, encoder_model, basis_model, baseline_model, logger)

    # train trial
    trainer.train(
        N_ITERS,
        UPDATE_EVERY,
        N_EPISODES,
        SAMPLE_BATCH_SIZE,
        SKILL_EPOCHS,
        PI_EPOCHS,
        LR,
        BATCH_SIZE,
        DISCOUNT,
        ALPHA_ENTROPY,
        SMOOTHING
    )


if __name__ == '__main__':
    main()