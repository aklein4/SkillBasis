
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import math

from utils import DEVICE, np2torch, torch2np


PPO_CLIP = 0.1


class Trainer:
    def __init__(self, env, baseline, epi_model, pi_model, logger=None):
        """ Trainer for the platform environment.

        Args:
            env (gym.env): Platform environment.
            q_model (QModel): Q network to train
            pi_model (PiModel): Parameter network to train
            logger (Logger, optional): Logger to save information. Defaults to None.
        """

        self.env = env
        self.baseline = baseline
        self.epi_model = epi_model
        self.pi_model = pi_model
        self.logger = logger

        # hyperparameters set in train call
        self.discount = None
        self.smoothing = None


    def _smooth(self, accum, next):
        if accum is None:
            return next
        return self.smoothing * accum + (1 - self.smoothing) * next
    

    def _loss(self, batch):

        # get Q values for current state
        V = self.baseline(batch.states).squeeze(-1)
        baseline_loss = F.mse_loss(V, batch.returns)
        
        # handle pi_g
        pi_G = self.pi_model(batch.states)

        p_G = torch.exp(pi_G.log_prob(batch.actions.unsqueeze(-1)))
        max_g, min_g = torch.max(p_G, dim=-1)[0], torch.min(p_G, dim=-1)[0]
        p_best = torch.where(batch.advantages >= 0, max_g, min_g)
        ratio_best = p_best / batch.og_probs

        G_loss = -torch.mean(
            torch.min(
                ratio_best * batch.advantages,
                torch.clamp(ratio_best, 1 - PPO_CLIP, 1 + PPO_CLIP) * batch.advantages
            )
        )

        # handle epi
        epi = self.epi_model(batch.states)

        p = torch.sum(epi.probs * p_G.detach(), dim=-1)
        ratio = p / batch.og_probs

        epi_loss = -torch.mean(
            torch.min(
                ratio * batch.advantages,
                torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * batch.advantages
            )
        )
        
        # handle prime pi
        pi_prime = self.pi_model(batch.states, disable_g=True)

        p_prime = torch.exp(pi_prime.log_prob(batch.actions))
        ratio_prime = p_prime / batch.og_probs
        
        prime_loss = -torch.mean(
            torch.min(
                ratio_prime * batch.advantages,
                torch.clamp(ratio_prime, 1 - PPO_CLIP, 1 + PPO_CLIP) * batch.advantages
            )
        )
        
        # get KL penalty
        p_targ = pi_prime.probs.unsqueeze(-2).detach()
        kl_raw = torch.mean(torch.sum(pi_G.probs * torch.log(pi_G.probs / p_targ), dim=-1), dim=-2)
        kl_loss  = 10*torch.mean(kl_raw)
        kl_log = torch.sum(kl_raw).item()

        return (
            baseline_loss + G_loss + epi_loss + prime_loss + kl_loss,
            kl_log
        )


    def train(self,
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
        self.discount = discount
        self. env.discount = discount
        self.smoothing = smoothing

        # things that are tracked for logging
        rolling_returns = None
        rolling_switch_perc = None
        rolling_skill_len = None
        rolling_kl = None

        # initialize optimizers
        baseline_opt = torch.optim.AdamW(self.baseline.parameters(), lr=lr)
        epi_opt = torch.optim.AdamW(self.epi_model.parameters(), lr=lr)
        pi_opt = torch.optim.AdamW(self.pi_model.parameters(), lr=lr)

        # run for iterations
        pbar = tqdm(range(num_iters), desc='Training', leave=True)
        for it in pbar:

            # get samples
            buffer, returns = self.env.sample(episodes_per_iter)

            # handle the rolling metrics
            rolling_returns = self._smooth(rolling_returns, sum(returns) / len(returns))
            rolling_switch_perc = self._smooth(rolling_switch_perc, buffer.get_switch_perc())
            rolling_skill_len = self._smooth(rolling_skill_len, buffer.get_avg_skill_len())

            self.baseline.eval()

            # calculate the adjusted advantages
            for i in range(0, len(buffer), batch_size):
                batch = buffer[(i, batch_size)]
                V = self.baseline(batch.states).squeeze(-1)
                batch.advantages[:] = batch.returns - V
            mu_A = torch.mean(buffer.advantages)
            std_A = torch.std(buffer.advantages)
            buffer.advantages[:] = (buffer.advantages - mu_A) / std_A
            buffer.advantages.detach_()

            # set model modes
            self.baseline.train()
            self.pi_model.train()
            self.epi_model.train()

            accum_kl = 0
            # train for epochs
            for epoch in range(epochs_per_iter):

                # sample vals from buffer
                buffer.shuffle()
                for i in range(0, len(buffer), batch_size):
                    batch = buffer[(i, batch_size)]

                    # training step
                    baseline_opt.zero_grad()
                    epi_opt.zero_grad()
                    pi_opt.zero_grad()

                    loss, kl = self._loss(batch)
                    accum_kl += kl

                    loss.backward()

                    baseline_opt.step()
                    epi_opt.step()
                    pi_opt.step()

            rolling_kl = self._smooth(rolling_kl, accum_kl / len(buffer))

            # log things
            if self.logger is not None:
                self.logger.log(
                    rolling_returns,
                    rolling_kl,
                    rolling_switch_perc,
                    rolling_skill_len,
                    torch2np(buffer.genes).tolist()
                )

            # update progress bar
            pbar.set_postfix({
                'iter': it,
                'r': round(rolling_returns, 3),
                'kl': round(rolling_kl, 3),
                'switch': round(rolling_switch_perc, 3),
                'skill_len': round(rolling_skill_len, 3)
            })
        
            if self.logger is not None and it % self.logger.save_every == 0:
                self.logger.save(self.epi_model, self.pi_model)
        
        # save models
        if self.logger is not None:
            self.logger.save(self.epi_model, self.pi_model)