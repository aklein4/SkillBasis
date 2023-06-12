
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from models import Baseline

from tqdm import tqdm


BASELINE_LR_COEF = 1

REG_L2 = 1e-3
REG_ENTROPY = 0.1

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
        self.alpha_entropy = None

        # target model for baseline
        self.target_model = Baseline().to(utils.DEVICE)
        self._update_target()

    
    def _update_target(self):
        self.target_model.load_state_dict(self.baseline_model.state_dict())


    def _smooth(self, accum, next):
        if accum is None:
            return next
        return self.smoothing * accum + (1 - self.smoothing) * next
    

    def _get_z_log_prob(self, batch, L_norm_too=False):
        # get state encodings
        l = self.encoder_model(batch.states)
        l_next = self.encoder_model(batch.next_states)
        delta_l = l_next - l

        # get skills basis
        L, L_norm = self.basis_model(len(batch))

        # project the transition into the skill space
        proj = torch.bmm(L, delta_l.unsqueeze(-1)).squeeze(-1)

        # get log_probs of each skill
        logmoid = torch.log(torch.sigmoid(batch.z_vals * proj))

        # weighted sum the logmoids to get total log probability
        z_log_prob = torch.sum(batch.z_attns * logmoid, dim=-1)

        if L_norm_too:
            return z_log_prob, L_norm
        return z_log_prob


    def _skill_loss(self, batch):

        # get log probabilities of each skill
        z_log_prob, L_norm = self._get_z_log_prob(batch, True)

        # want to maximize the log probability of the skills
        z_loss = -torch.mean(z_log_prob)

        # add L2 regularization to the basis
        norm_loss = REG_L2 * L_norm

        return z_loss + norm_loss


    def _pi_loss(self, batch):

        """ Training """

        # weighted sum the logmoids to get total log probability
        z_log_prob = self._get_z_log_prob(batch).detach()

        # get the policy
        pi = self.pi_model(batch.states, batch.skills)
        pi_log_probs = pi.log_prob(batch.actions)

        # get the entropy of the next state
        pi_next = pi.entropy()
        pi_entropy_next = pi_next.entropy().detach()

        # get the reward
        reward = z_log_prior + self.alpha_entopy * pi_entropy_next
        reward = reward.detach()

        # get baselines
        V = self.baseline_model(batch.states, batch.skills).squeeze(-1)
        V_next = self.target_model(batch.next_states, batch.skills).squeeze(-1).detach()
        value = reward + self.discount * V_next

        # use baseline
        baseline_loss = F.mse_loss(V, value)
        advantage = (value - V).unsqueeze(-1).detach()

        # get importance sampling ratio
        pi_probs = torch.exp(pi_log_probs)
        ratio = pi_probs / batch.og_probs

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

        # calculate the mutual information
        z_log_prior = self.env.skill_generator.log_prob(len(batch))
        mutual_info = z_log_prob - z_log_prior

        # get logging metrics
        logging_mutual_info = torch.sum(mutual_info).item()
        logging_baseline_loss = len(batch)*baseline_loss.item()
        logging_entropy = torch.sum(pi_entropy_next).item()

        # get total loss
        return pi_loss + baseline_loss, logging_mutual_info, logging_baseline_loss, logging_entropy


    def train(self,
            n_iters,
            update_every,
            n_episodes,
            z_epochs,
            pi_epochs,
            lr,
            batch_size,
            discount,
            alpha_entropy,
            smoothing
        ):

        # set hyperparameters
        self.smoothing = smoothing
        self.discount = discount
        self.alpha_entropy = alpha_entropy

        # things that are tracked for logging
        rolling_mutual = None
        rolling_baseline_loss = None
        rolling_entropy = None

        # initialize optimizers
        pi_opt = torch.optim.AdamW(self.pi_model.parameters(), lr=lr)
        enc_opt = torch.optim.AdamW(self.encoder_model.parameters(), lr=lr)
        basis_opt = torch.optim.AdamW(self.basis_model.parameters(), lr=lr)
        baseline_opt = torch.optim.AdamW(self.baseline_model.parameters(), lr=lr*BASELINE_LR_COEF)

        # run for iterations
        pbar = tqdm(range(n_iters), desc='Training', leave=True)
        for it in pbar:

            # get samples
            buffer = self.env.sample(n_episodes)

            """ ----- Train Skills ----- """

            # set model modes
            self.encoder_model.train()
            self.basis_model.train()

            # train for epochs
            for epoch in range(z_epochs):
                buffer.shuffle()
                for i in range(0, len(buffer), batch_size):
                    batch = buffer[(i, batch_size)]

                    enc_opt.zero_grad()
                    basis_opt.zero_grad()

                    loss = self._skill_loss(batch)

                    loss.backward()
                    enc_opt.step()
                    basis_opt.step()

            """ ----- Train Policy ----- """

            # set model modes
            self.pi_model.train()
            self.encoder_model.train()
            self.basis_model.eval()
            self.baseline_model.eval()
            self.target_model.eval()

            # accumulate logging info
            mutual_accum = 0
            baseline_loss_accum = 0
            entropy_accum = 0

            # train for epochs
            for epoch in range(pi_epochs):
                buffer.shuffle()
                for i in range(0, len(buffer), batch_size):
                    batch = buffer[(i, batch_size)]

                    pi_opt.zero_grad()
                    baseline_opt.zero_grad()

                    loss, mutual, baseline_loss, entropy = self._pi_loss(batch)

                    loss.backward()

                    pi_opt.step()
                    baseline_opt.step()

                    mutual_accum += mutual
                    baseline_loss_accum += baseline_loss
                    entropy_accum += entropy

            # handle metrics
            rolling_mutual = self._smooth(rolling_mutual, mutual_accum/len(buffer))
            rolling_baseline_loss = self._smooth(rolling_baseline_loss, baseline_loss_accum/len(buffer))
            rolling_entropy = self._smooth(rolling_entropy, entropy_accum/len(buffer))

            # log things
            if self.logger is not None:
                self.logger.log(rolling_mutual, rolling_baseline_loss, rolling_entropy)

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