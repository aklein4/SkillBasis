
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from models import Baseline

from tqdm import tqdm


BASELINE_LR_COEF = 1

REG_L2 = 1e-3


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

        self.target_model = Baseline()

    
    def _update_target(self):
        self.target_model.load_state_dict(self.baseline_model.state_dict())


    def _smooth(self, accum, next):
        if accum is None:
            return next
        return self.smoothing * accum + (1 - self.smoothing) * next
    

    def _loss(self, batch):

        # get state encodings
        l = self.encoder_model(batch.next_states)

        # get skills basis
        L, z_sigmas = self.basis_model(batch.states)
        # normalize basis
        L_reg = torch.mean( torch.abs(1 - torch.norm(L, p=2, dim=-1) ) )
        L = L / torch.norm(L, p=2, dim=-1, keepdim=True).detach()

        # get predicted skill
        z_mus = torch.bmm(L, l.unsqueeze(-1)).squeeze(-1)
        z_dist = torch.distributions.Normal(z_mus, z_sigmas)

        # get prob of actual skill
        z_log_prob = torch.sum(z_dist.log_prob(batch.skills), dim=-1)
        z_loss = -torch.mean(z_log_prob)

        # get z prior and reward
        z_log_prior = torch.sum(self.env.skill_generator.log_prob(batch.skills), dim=-1)
        reward = z_log_prob.detach() - z_log_prior

        # get baselines
        V = self.baseline_model(batch.states, batch.skills).squeeze(-1)
        V_next = self.target_model(batch.next_states, batch.skills).squeeze(-1).detach()
        value = reward + self.discount * V_next

        # use baseline
        baseline_loss = F.mse_loss(V, value)
        advantage = value - V.detach()

        # get pi error
        pi = self.pi_model(batch.states, batch.skills)
        log_probs = pi.log_prob(batch.actions)
        if log_probs.dim() > 1:
            log_probs = torch.sum(log_probs, dim=-1)
        pi_loss = -torch.mean(log_probs * advantage)

        # get total loss
        return pi_loss + z_loss + baseline_loss + REG_L2*L_reg, torch.sum(reward).item(), len(batch)*baseline_loss.item()


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
                'buf': len(buffer)
            })
        
        # save models
        if self.logger is not None:
            self.logger.save()