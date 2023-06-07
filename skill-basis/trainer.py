
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from models import Baseline

from tqdm import tqdm
import math


BASELINE_LR_COEF = 1

REG_L2 = 1e-3
REG_ENTROP = 0.0

LOG_CLIP = -7
PPO_CLIP = 0.2

class Trainer:
    def __init__(self,
            env,
            pi_model,
            encoder_model,
            basis_model,
            baseline_model,
            logger=None
        ):

        self.env = env

        self.pi_model = pi_model
        self.encoder_model = encoder_model
        self.basis_model = basis_model
        self.baseline_model = baseline_model

        self.logger = logger

        # hyperparameters set in train call
        self.smoothing = None
        self.discount = None

        self.target_model = Baseline().to(utils.DEVICE)

    
    def _update_target(self):
        self.target_model.load_state_dict(self.baseline_model.state_dict())


    def _smooth(self, accum, next):
        if accum is None:
            return next
        return self.smoothing * accum + (1 - self.smoothing) * next
    

    def _loss(self, batch):

        # get state encodings
        l = self.encoder_model(batch.states)
        l_next = self.encoder_model(batch.next_states)
        delta_l = l_next - l

        # get skills basis
        L, z_sigmas = self.basis_model(len(batch))
        # normalize basis
        L_reg = torch.mean( torch.abs(1 - torch.norm(L, p=2, dim=-1) ) )
        L = L / torch.norm(L, p=2, dim=-1, keepdim=True).detach()

        # get predicted skill
        z_mus = torch.bmm(L, delta_l.unsqueeze(-1)).squeeze(-1)
        z_dist = torch.distributions.Normal(z_mus, z_sigmas)

        # get prob of actual skill
        z_log_prob = torch.sum(z_dist.log_prob(batch.skills), dim=-1)
        z_log_prob = torch.clamp(z_log_prob, min=LOG_CLIP, max=0)
        z_loss = -torch.mean(z_log_prob)

        # get z prior and reward
        z_log_prior = torch.sum(self.env.skill_generator.log_prob(batch.skills), dim=-1)
        z_log_prior = torch.clamp(z_log_prior, min=LOG_CLIP, max=0)
        reward = z_log_prob.detach() - z_log_prior

        # get baselines
        V = self.baseline_model(batch.states, batch.skills).squeeze(-1)
        V_next = self.target_model(batch.next_states, batch.skills).squeeze(-1).detach()
        value = reward + self.discount * V_next

        # use baseline
        baseline_loss = F.mse_loss(V, value)
        advantage = (value - V.detach()).unsqueeze(-1)

        # get pi
        pi = self.pi_model(batch.states, batch.skills)
        log_probs = pi.log_prob(batch.actions)
        probs = torch.exp(log_probs)
        ratio = probs / torch.clamp(batch.og_probs, min=math.exp(LOG_CLIP), max=1)
        ratio = torch.nan_to_num(ratio, nan=1, posinf=1, neginf=1)

        # get pi loss
        pi_loss = -torch.mean(
            torch.sum(
                torch.min(
                    ratio * advantage,
                    torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantage
                ),
                dim=-1
            )
        )

        entrop_loss = -torch.mean(torch.sum(log_probs, dim=-1))

        # get total loss
        return pi_loss + z_loss + baseline_loss + REG_L2*L_reg + REG_ENTROP*entrop_loss, torch.sum(reward).item(), len(batch)*baseline_loss.item()


    def train(self,
            num_iters,
            update_every,
            episodes_per_iter,
            epochs_per_iter,
            lr,
            batch_size,
            discount,
            smoothing
        ):
        self.smoothing = smoothing
        self.discount = discount
        self._update_target()

        # things that are tracked for logging
        logging_loss = None
        logging_baseline_loss = None

        # initialize optimizers
        pi_opt = torch.optim.AdamW(self.pi_model.parameters(), lr=lr)
        enc_opt = torch.optim.AdamW(self.encoder_model.parameters(), lr=lr)
        basis_opt = torch.optim.AdamW(self.basis_model.parameters(), lr=lr)
        baseline_opt = torch.optim.AdamW(self.baseline_model.parameters(), lr=lr*BASELINE_LR_COEF)

        # run for iterations
        pbar = tqdm(range(num_iters), desc='Training', leave=True)
        for it in pbar:

            # get samples
            buffer = self.env.sample(episodes_per_iter)

            # set model modes
            self.pi_model.train()
            self.encoder_model.train()
            self.basis_model.train()
            self.baseline_model.train()
            self.target_model.eval()

            # accumulate logging info
            this_loss = 0
            this_baseline_loss = 0

            # train for epochs
            for epoch in range(epochs_per_iter):

                # sample vals from buffer
                buffer.shuffle()
                for i in range(0, len(buffer), batch_size):
                    batch = buffer[(i, batch_size)]

                    # training step
                    pi_opt.zero_grad()
                    enc_opt.zero_grad()
                    basis_opt.zero_grad()
                    baseline_opt.zero_grad()

                    loss, to_log, to_baseline_log = self._loss(batch)

                    loss.backward()

                    pi_opt.step()
                    enc_opt.step()
                    basis_opt.step()
                    baseline_opt.step()

                    this_loss += to_log
                    this_baseline_loss += to_baseline_log

            # handle metrics
            logging_loss = self._smooth(logging_loss, this_loss/len(buffer))
            logging_baseline_loss = self._smooth(logging_baseline_loss, this_baseline_loss/len(buffer))

            # log things
            if self.logger is not None:
                self.logger.log(logging_loss, logging_baseline_loss)

            if it+1 % update_every == 0:
                self._update_target()

            # update progress bar
            pbar.set_postfix({
                'iter': it,
                'r': round(logging_loss, 3),
            })
        
        # save models
        if self.logger is not None:
            self.logger.save()