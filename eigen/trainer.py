
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import math

from utils import DEVICE, np2torch, torch2np


PPO_CLIP = 0.2


class Trainer:
    def __init__(self, env, pi_model, baseline, logger=None):
        """ Trainer for the platform environment.

        Args:
            env (gym.env): Platform environment.
            q_model (QModel): Q network to train
            pi_model (PiModel): Parameter network to train
            logger (Logger, optional): Logger to save information. Defaults to None.
        """

        self.env = env

        self.baseline = baseline
        self.pi_model = pi_model

        self.logger = logger

        # hyperparameters set in train call
        self.smoothing = None


    def _smooth(self, accum, next):
        if accum is None:
            return next
        return self.smoothing * accum + (1 - self.smoothing) * next
    

    def _loss(self, batch):

        # get Q values for current state
        V = self.baseline(batch.states, batch.modes).squeeze(-1)
        baseline_loss = F.mse_loss(V, batch.returns)
        
        advantages = batch.returns - V.detach()

        # handle pi_g
        pi = self.pi_model(batch.states, batch.modes)
        probs = torch.exp(pi.log_prob(batch.actions))
        ratio_best = probs / batch.og_probs

        pi_loss = -torch.mean(
            torch.min(
                ratio_best * advantages,
                torch.clamp(ratio_best, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages
            )
        )

        return baseline_loss + pi_loss


    def train(self,
              modes,
              num_iters,
              episodes_per_iter,
              epochs_per_iter,
              lr,
              batch_size,
              discount,
              smoothing):
        """ Train the models.

        Args:
            num_iters (int): Number of training iterations
            epsilon_decay_iters (int): Number of iterations for epsilon to zero
            iters_per_target (int): Number of iterations to update target networks
            episodes_per_iter (int): Number of episodes to sample per iteration
            epochs_per_iter (int): Number of epochs to train per iteration
            q_vals_per_epoch (int): Number of values to sample from replay buffer per epoch
            buffer_size (int): replay buffer size
            lr (float): learning rate
            batch_size (int): batch size
            discount (int, optional): discount coefficient. Defaults to 1.
            p_momentum (float, optional): Momentum for q_loss weight norm. Defaults to 0.98.
            smoothing (float, optional): Momentum for logging rewards. Defaults to 0.
        """

        """ ----- Set up ----- """

        # initialize hyperparameters
        self.env.discount = discount
        self.smoothing = smoothing

        # things that are tracked for logging
        rolling_returns = [None for _ in range(len(modes))]

        # initialize optimizers
        baseline_opt = torch.optim.AdamW(self.baseline.parameters(), lr=lr)
        pi_opt = torch.optim.AdamW(self.pi_model.parameters(), lr=lr)

        # run for iterations
        pbar = tqdm(range(num_iters), desc='Training', leave=True)
        for it in pbar:

            # get samples
            buffer = None
            for mode in modes:
                new_buf, returns = self.env.sample(episodes_per_iter, mode)
                buffer = new_buf if buffer is None else buffer + new_buf
                rolling_returns[mode] = self._smooth(rolling_returns[mode], sum(returns) / len(returns))

            # set model modes
            self.baseline.train()
            self.pi_model.train()

            # train for epochs
            for epoch in range(epochs_per_iter):

                # sample vals from buffer
                buffer.shuffle()
                for i in range(0, len(buffer), batch_size):
                    batch = buffer[(i, batch_size)]

                    # training step
                    baseline_opt.zero_grad()
                    pi_opt.zero_grad()

                    loss = self._loss(batch)

                    loss.backward()

                    baseline_opt.step()
                    pi_opt.step()

                    self.baseline.normalize()
                    self.pi_model.normalize()

            # log things
            if self.logger is not None:
                self.logger.log(rolling_returns)

            # update progress bar
            pbar.set_postfix({
                'iter': it,
                'r': round(sum(rolling_returns)/len(rolling_returns), 3),
            })
        
        # save models
        if self.logger is not None:
            self.logger.save()