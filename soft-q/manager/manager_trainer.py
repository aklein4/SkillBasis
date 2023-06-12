
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from models import ManagerBaseline

from tqdm import tqdm
import math


BASELINE_LR_COEF = 1

LOG_CLIP = -7
PPO_CLIP = 0.2

class Trainer:
    def __init__(self,
            env,
            man_model,
            baseline_model,
            logger=None
        ):

        self.env = env

        self.man_model = man_model
        self.baseline_model = baseline_model

        self.logger = logger

        # hyperparameters set in train call
        self.smoothing = None
        self.discount = None

        self.target_model = ManagerBaseline().to(utils.DEVICE)

    
    def _update_target(self):
        self.target_model.load_state_dict(self.baseline_model.state_dict())


    def _smooth(self, accum, next):
        if accum is None:
            return next
        return self.smoothing * accum + (1 - self.smoothing) * next
    

    def _loss(self, batch):

        # get baselines
        V = self.baseline_model(batch.states, batch.skills).squeeze(-1)
        V_next = self.target_model(batch.next_states, batch.skills).squeeze(-1).detach()
        value = batch.rewards + self.discount * V_next
        value = torch.where(batch.dones, batch.rewards, value)

        # use baseline
        baseline_loss = F.mse_loss(V, value)
        advantage = (value - V.detach()).unsqueeze(-1)

        # get pi
        pi = self.man_model(batch.states)
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

        # get total loss
        return pi_loss + baseline_loss, len(batch)*baseline_loss.item()


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
        man_opt = torch.optim.AdamW(self.man_model.parameters(), lr=lr)
        baseline_opt = torch.optim.AdamW(self.baseline_model.parameters(), lr=lr*BASELINE_LR_COEF)

        # run for iterations
        pbar = tqdm(range(num_iters), desc='Training', leave=True)
        for it in pbar:

            # get samples
            buffer, rewards = self.env.sample(episodes_per_iter)
            rolling_loss = self._smooth(rolling_loss, sum(rewards)/len(rewards))

            # set model modes
            self.man_model.train()
            self.baseline_model.train()
            self.target_model.eval()

            # accumulate logging info
            this_baseline_loss = 0

            # train for epochs
            for epoch in range(epochs_per_iter):

                # sample vals from buffer
                buffer.shuffle()
                for i in range(0, len(buffer), batch_size):
                    batch = buffer[(i, batch_size)]

                    # training step
                    man_opt.zero_grad()
                    baseline_opt.zero_grad()

                    loss, to_baseline_log = self._loss(batch)

                    loss.backward()

                    man_opt.step()
                    baseline_opt.step()

                    this_baseline_loss += to_baseline_log

            # handle metrics
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
                'b': round(logging_baseline_loss, 3)
            })
        
        # save models
        if self.logger is not None:
            self.logger.save()