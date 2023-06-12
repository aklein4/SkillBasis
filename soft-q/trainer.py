
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from models import Encoder, Policy

from tqdm import tqdm


REG_L2 = 0
REG_DIST = 0

L_SCALE = 10
REWARD_SCALE = 10

CLAM = 1e-3


class Trainer:
    def __init__(self,
            env,
            pi_model,
            encoder_model,
            basis_model,
            logger=None
        ):

        self.env = env

        self.pi_model = pi_model
        self.encoder_model = encoder_model
        self.basis_model = basis_model

        self.logger = logger

        # hyperparameters set in train call
        self.smoothing = None
        self.discount = None

        # target model for baseline
        self.target_pi = Policy().to(utils.DEVICE)
        self.target_encoder = Encoder().to(utils.DEVICE)
        self._update_target()

    
    def _update_target(self):
        self.target_pi.load_state_dict(self.pi_model.state_dict())
        self.target_encoder.load_state_dict(self.encoder_model.state_dict())


    def _smooth(self, accum, next):
        if accum is None:
            return next
        return self.smoothing * accum + (1 - self.smoothing) * next
    

    def _get_z_log_prob(self, batch, L_norm_too=False, use_delta=True):

        # get state encodings
        l = self.encoder_model(batch.states).detach()
        l_next = self.encoder_model(batch.next_states)

        delta_l = (l_next - l) * L_SCALE

        # get skills basis
        L, L_norm = self.basis_model(len(batch))

        # project the transition into the skill space
        proj = torch.bmm(L, delta_l.unsqueeze(-1)).squeeze(-1)

        # get log_probs of each skill
        logmoid = torch.log(
            torch.clamp(torch.sigmoid(batch.z_vals * proj), min=CLAM)
        )

        # weighted sum the logmoids to get total log probability
        z_log_prob = torch.sum(batch.z_attns * logmoid, dim=-1)

        if L_norm_too:
            return z_log_prob, L_norm, torch.norm(delta_l, p=2, dim=-1)
        return z_log_prob


    def _get_mutual_info(self, batch):
        z_log_prob = self._get_z_log_prob(batch, use_delta=False)
        z_log_prior = self.env.skill_generator.log_prob(len(batch))
        return z_log_prob - z_log_prior
    

    def _skill_loss(self, batch):

        # get log probabilities of each skill
        z_log_prob, L_norm, dist = self._get_z_log_prob(batch, True)

        # want to maximize the log probability of the skills
        z_loss = -torch.mean(z_log_prob)

        # add L2 regularization to the basis
        norm_loss = REG_L2 * L_norm

        # add L2 regularization to the distance between state encodings
        dist_loss = torch.mean(REG_DIST * dist)

        return z_loss + norm_loss + dist_loss


    def _pi_loss(self, batch):

        # get the mutual information
        mutual_info = self._get_mutual_info(batch)

        # get the reward
        reward = mutual_info
        reward = reward.detach() * REWARD_SCALE

        # get model predictions
        Q = self.pi_model.Q(batch.states, batch.z_vals, batch.z_attns)
        Q_a = Q[range(len(batch)), batch.actions]
        V_next = self.target_pi.V(batch.next_states, batch.z_vals, batch.z_attns).detach()

        # get the target value
        target = (reward + self.discount * V_next).detach()

        # loss is bellman
        loss = F.mse_loss(Q_a, target)

        # get logging metrics
        logging_q_loss = 2 * torch.sum(torch.abs(target - Q_a)/(torch.abs(target) + torch.abs(Q_a))).item()
        logging_entropy = torch.sum(self.pi_model.entropy(Q)).item()

        # get total loss
        return loss, logging_q_loss, logging_entropy


    def train(self,
            n_iters,
            update_every,
            n_episodes,
            sample_batch_size,
            z_epochs,
            pi_epochs,
            lr,
            batch_size,
            buffer_size,
            discount,
            smoothing
        ):

        # set hyperparameters
        self.smoothing = smoothing
        self.discount = discount

        # things that are tracked for logging
        rolling_mutual = None
        rolling_q_loss = None
        rolling_entropy = None

        # initialize optimizers
        pi_opt = torch.optim.AdamW(self.pi_model.parameters(), lr=lr)
        enc_opt = torch.optim.AdamW(self.encoder_model.parameters(), lr=lr)
        basis_opt = torch.optim.AdamW(self.basis_model.parameters(), lr=lr)

        # replay buffer to avoid catastrophic forgetting
        buffer = None

        # run for iterations
        pbar = tqdm(range(n_iters), desc='Training', leave=True)
        for it in pbar:

            # get samples
            new_buffer = self.env.sample(n_episodes, batch_size=sample_batch_size)

            # get the reward for the episode
            self.encoder_model.eval()
            self.basis_model.eval()
            self.target_encoder.eval()
            rolling_mutual = self._smooth(rolling_mutual, torch.sum(self._get_mutual_info(new_buffer)).item()/len(new_buffer))

            """ ----- Train Skills ----- """

            # set model modes
            self.encoder_model.train()
            self.basis_model.train()
            self.target_encoder.eval()

            # train for epochs
            for epoch in range(z_epochs):
                new_buffer.shuffle()
                for i in range(0, len(new_buffer), batch_size):
                    batch = new_buffer[(i, batch_size)]

                    enc_opt.zero_grad()
                    basis_opt.zero_grad()

                    loss = self._skill_loss(batch)

                    loss.backward()
                    enc_opt.step()
                    basis_opt.step()

            """ ----- Train Policy ----- """

            # set model modes
            self.pi_model.train()

            self.encoder_model.eval()
            self.basis_model.eval()
            self.target_pi.eval()
            self.target_encoder.eval()

            # accumulate logging info
            q_loss_accum = 0
            entropy_accum = 0

            # combine into buffer
            if buffer is None:
                buffer = new_buffer
            else:
                buffer = buffer + new_buffer

            # train for epochs
            for epoch in range(pi_epochs):
                buffer.shuffle()
                for i in range(0, len(buffer), batch_size):
                    batch = buffer[(i, batch_size)]

                    pi_opt.zero_grad()

                    loss, q_loss, entropy = self._pi_loss(batch)

                    loss.backward()

                    pi_opt.step()

                    q_loss_accum += q_loss / pi_epochs
                    entropy_accum += entropy / pi_epochs

            # handle metrics
            rolling_q_loss = self._smooth(rolling_q_loss, q_loss_accum/len(buffer))
            rolling_entropy = self._smooth(rolling_entropy, entropy_accum/len(buffer))

            # reduce the buffer
            buffer.reduce(buffer_size)

            # log things
            if self.logger is not None:
                self.logger.log(rolling_mutual, rolling_q_loss, rolling_entropy)

            # update target model
            if it+1 % update_every == 0:
                self._update_target()

            # update progress bar
            pbar.set_postfix({
                'iter': it,
                'r': round(rolling_mutual, 3),
            })
        
        # save models
        if self.logger is not None:
            self.logger.save()