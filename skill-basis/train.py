
import torch

from environment import Environment, SkillGenerator
from models import Policy, Encoder, Basis, Baseline
from trainer import Trainer
from logger import Logger
from drone import Drone
import utils

# result directory
OUTPUT_DIR = 'test'
SAVE_EVERY = 32

# training parameters
NUM_ITERS = 2048 # number of training iterations
UPDATE_EVERY = 8

EPISODES_PER_ITER = 8 # episodes sampled per iteration
EPOCHS_PER_ITER = 1 # epochs trained per iteration

LR = 1e-4 # learning rate
BATCH_SIZE = 16 # batch size

DISCOUNT = 0.75

REWARD_SMOOTHING = 0.9 # reward logging momentum

SKILL_SIGMA = 1


def main():

    # fresh models
    pi_model = Policy()
    pi_model = pi_model.to(utils.DEVICE)
    # pi_model.load_state_dict(torch.load('test/pi_model.pt', map_location='cpu'))

    encoder_model = Encoder()
    encoder_model = encoder_model.to(utils.DEVICE)
    # encoder_model.load_state_dict(torch.load('test/encoder_model.pt', map_location='cpu'))

    basis_model = Basis()
    basis_model = basis_model.to(utils.DEVICE)
    # basis_model.load_state_dict(torch.load('test/basis_model.pt', map_location='cpu'))

    baseline_model = Baseline()
    baseline_model = baseline_model.to(utils.DEVICE)
    # baseline_model.load_state_dict(torch.load('test/baseline_model.pt', map_location='cpu'))

    logger = Logger(pi_model, encoder_model, basis_model, baseline_model, OUTPUT_DIR, SAVE_EVERY)

    rocket = Drone(discrete=False)
    skill_gen = SkillGenerator(pi_model.config.n_skills)
    env = Environment(rocket, pi_model, skill_gen)

    trainer = Trainer(env, pi_model, encoder_model, basis_model, baseline_model, logger)

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