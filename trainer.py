
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import math

from models import EpiPolicy, POLICY_MODES
from utils import DEVICE


PPO_CLIP = 0.2


class Trainer:
    def __init__(self, env, q_model, pi_model, logger=None):
        """ Trainer for the platform environment.

        Args:
            env (gym.env): Platform environment.
            q_model (QModel): Q network to train
            pi_model (PiModel): Parameter network to train
            logger (Logger, optional): Logger to save information. Defaults to None.
        """

        self.env = env
        self.q_model = q_model
        self.pi_model = pi_model
        self.logger = logger

        # target models
        self.q_target = EpiPolicy()
        self.q_target = self.q_target.to(DEVICE)

        # hyperparameters set in train call
        self.discount = None
        self.p_norm = None # rolling norm for q loss
        self.p_momentum = None # momentum for rolling_p
        self.epsilon = None
        self.switch_penalty = None


    def _update_target(self):
        """
        Update target networks with current network weights.
        """
        self.q_target.load_state_dict(self.q_model.state_dict()) 


    def _q_loss(self, batch):

        # get Q values
        Q = self.q_model(batch.states)
        Q_g = Q[range(len(batch)), batch.genes]
        Q_target = self.q_target(batch.next_states).detach()

        target = batch.rewards + self.discount*torch.max(Q_target, dim=-1).values

        pi_g = self.pi_model(batch.states, batch.genes)
        importance = torch.exp(pi_g.log_prob(batch.actions)).detach()

        if self.p_norm is None:
            self.p_norm = importance.mean().item()
        else:
            self.p_norm = self.p_momentum * self.p_norm + (1 - self.p_momentum) * importance.mean().item()
        for i in range(len(batch)):
            batch.importance[i] = importance[i] / self.p_norm

        loss = torch.sum(importance/self.p_norm * (Q_g - target)**2)

        # more stable version for logging (mean abs RPD)
        logging_loss = 2 * torch.sum(torch.abs(Q_g - target) / (torch.abs(Q_g) + torch.abs(target))).item()

        # variance for logging
        var = torch.sum(torch.std(Q, dim=-1)).item()

        return loss, logging_loss, var
    

    def _pi_loss(self, batch, g_grad):

        Q = self.q_model(batch.states).detach()
        adj_Q = Q.clone()
        adj_Q[range(len(batch)), batch.prev_genes] -= self.switch_penalty

        baseline = Q[range(len(batch)), torch.argmax(adj_Q, dim=-1)]
        if not g_grad:
            g_probs = torch.full_like(Q, self.epsilon/Q.shape[-1])
            g_probs[range(len(batch)), torch.argmax(adj_Q, dim=-1)] += 1 - self.epsilon
            baseline = torch.sum(g_probs * Q, dim=-1)

        Q_next = self.q_model(batch.next_states).detach()
        G = batch.rewards + self.discount*torch.max(Q_next, dim=-1).values

        advantage = G - baseline

        pi = self.pi_model(batch.states, torch.argmax(adj_Q, dim=-1))
        ratio = torch.exp(pi.log_prob(batch.actions)) / batch.og_probs

        loss = -torch.sum(
            torch.min(
                ratio * advantage,
                torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantage
            )
        )

        return loss


    def train(self,
              num_iters,
              epsilon_decay_iters,
              iters_per_target,
              episodes_per_iter,
              epochs_per_iter,
              q_vals_per_epoch,
              buffer_size,
              lr,
              batch_size,
              discount,
              switch_penalty,
              p_momentum,
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
        self.p_momentum = p_momentum
        self.switch_penalty = switch_penalty
        self.env.switch_penalty = switch_penalty

        # initialize vars
        self.epsilon = 1
        self.p_norm = None
        rolling_returns = None

        # initialize optimizers
        q_opt = torch.optim.AdamW(self.q_model.parameters(), lr=lr)
        pi_opt = torch.optim.AdamW(self.pi_model.parameters(), lr=lr)

        # fill the buffer with init vals
        self.pi_model.set_mode(POLICY_MODES.NORMAL)
        buffer, _ = self.env.sample(episodes_per_iter, self.epsilon)
        # pbar = tqdm(range(buffer_size), desc='Initializing Buffer', leave=False)
        # while len(buffer) < buffer_size:
        #     new_buffer, _ = self.env.sample(episodes_per_iter, self.epsilon)
        #     buffer = buffer + new_buffer
        #     pbar.update(len(new_buffer))

        # keep track of for logging
        kls = None

        # run for iterations
        pbar = tqdm(range(num_iters), desc='Training', leave=True)
        for it in pbar:

            """ -----  Buffer Sampling ----- """

            # update hyperparameters
            self.epsilon = max(0, 1 - it / epsilon_decay_iters)
            
            # sample new transitions
            self.pi_model.set_mode(POLICY_MODES.NORMAL)
            new_buffer, returns = self.env.sample(episodes_per_iter, self.epsilon)
            buffer = buffer + new_buffer

            # handle the returns
            avg_return = sum(returns) / len(returns)
            if rolling_returns is None:
                rolling_returns = avg_return
            else:
                rolling_returns = rolling_returns * smoothing + (1 - smoothing) * avg_return

            """-----  Q Training -----  """

            # set model modes
            self.q_model.train()
            self.pi_model.eval()
            self.pi_model.set_mode(POLICY_MODES.NORMAL)
            self.q_target.eval()

            # things that are accumulated then logged
            q_losses = 0
            q_vars = 0

            q_training_examples = min(len(buffer), q_vals_per_epoch)

            # train for epochs
            for epoch in range(epochs_per_iter):

                # sample vals from buffer
                buffer.shuffle()
                for i in range(0, q_training_examples, batch_size):
                    batch = buffer[(i, batch_size)]

                    # training step
                    q_opt.zero_grad()

                    loss, q_loss, var = self._q_loss(batch)

                    loss.backward()

                    q_opt.step()

                    # logged later
                    q_losses += q_loss
                    q_vars += var
            
            # average losses
            q_losses /= q_training_examples * epochs_per_iter
            q_vars /= q_training_examples * epochs_per_iter

            """-----  Main Training -----  """

            # set model modes
            self.q_model.eval()
            self.pi_model.train()
            self.pi_model.set_mode(POLICY_MODES.MAIN_ONLY)

            # train for epochs
            for epoch in range(epochs_per_iter):

                # sample vals from buffer
                new_buffer.shuffle()
                for i in range(0, len(new_buffer), batch_size):
                    batch = new_buffer[(i, batch_size)]

                    # training step
                    pi_opt.zero_grad()

                    loss = self._pi_loss(batch, False)

                    loss.backward()

                    pi_opt.step()

            """-----  Gene Training -----  """

            # set model modes
            self.pi_model.set_mode(POLICY_MODES.G_GRAD)

            # train for epochs
            for epoch in range(epochs_per_iter):

                # sample vals from buffer
                new_buffer.shuffle()
                for i in range(0, len(new_buffer), batch_size):
                    batch = new_buffer[(i, batch_size)]

                    # training step
                    pi_opt.zero_grad()

                    loss = self._pi_loss(batch, True)

                    loss.backward()

                    pi_opt.step()

            """-----  Logging -----  """

            # log things
            if self.logger is not None:
                self.logger.log(rolling_returns, q_losses, q_vars, math.log(self.p_norm), kls, self.epsilon)

            # update target network
            if it % iters_per_target == 0:
                self._update_target()

            if it == 0 or it % iters_per_target == 0:
                kls = 0
                self.pi_model.eval()
                self.pi_model.set_mode(POLICY_MODES.NORMAL)
                for i in range(0, len(buffer), batch_size):
                    batch = buffer[(i, batch_size)]
                    kls += self.pi_model.get_kl(batch.states)
                kls /= len(buffer)

            # remove bad samples and reduce buffer size
            buffer.reduce(buffer_size)

            # update progress bar
            pbar.set_postfix({
                'iter': it,
                'r': round(rolling_returns, 3),
                'q_loss': round(q_losses, 3),
                'q_var': round(q_vars, 3),
                'p_norm': math.log(self.p_norm),
                'kl': round(kls, 3),
                'buf': len(buffer),
            })
        
        # save models
        if self.logger is not None:
            self.logger.save(self.q_model, self.pi_model)