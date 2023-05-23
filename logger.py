
import torch

import os
import csv
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, log_loc):
        """ Logger to save training metrics.

        Args:
            log_loc (str): Folder for saving
        """

        # metrics to track
        self.returns = []
        self.q_losses = []
        self.q_vars = []
        self.p_norms = []
        self.kls = []
        self.epsilons = []

        # save location
        self.log_loc = log_loc

        # make folder
        os.makedirs(self.log_loc, exist_ok=True)
        
        # initialize metric csv
        with open(os.path.join(self.log_loc, "metrics.csv"), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow(["epoch", "return", "q_loss", "q_var", "p_norm", "kl", "epsilon"])


    def write(self):
        """
        Write latest metrics to csv.
        """
        with open(os.path.join(self.log_loc, "metrics.csv"), 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.returns)-1, self.returns[-1], self.q_losses[-1], self.q_vars[-1], self.p_norms[-1], self.kls[-1], self.epsilons[-1]])


    def plot(self):
        """
        Plot the metrics' progress.
        """

        # init figure
        fig, ax = plt.subplots(3, 2)

        # plot returns
        ax[0, 0].plot(self.returns)
        ax[0, 0].set_title("Average Returns (Smoothed)")
        ax[0, 0].set_ylabel('avg total reward')

        ax[0, 1].plot(self.epsilons)
        ax[0, 1].set_title("Epsilon")
        ax[0, 1].set_ylabel('epsilon')

        # plot q losses
        ax[1, 0].plot(self.q_losses)
        ax[1, 0].set_title("Q-Value Losses")
        ax[1, 0].set_ylabel('abs RPD') 

        # plot vars
        ax[1, 1].plot(self.q_vars)
        ax[1, 1].set_title("Inter-Gene Q-Value Standard Deviations")
        ax[1, 1].set_ylabel('avg sigma')

        # plot p_rollings
        ax[2, 0].plot(self.p_norms)
        ax[2, 0].set_title("Rolling Policy Probability Norms")
        ax[2, 0].set_ylabel('log p_norm')

        # plot p_rollings
        ax[2, 1].plot(self.kls)
        ax[2, 1].set_title("Inter-Gene Policy KL Divergences")
        ax[2, 1].set_ylabel('avg kl')

        # save figure to file
        fig.suptitle("Training Progress (Smoothed)")
        fig.tight_layout()
        fig.set_size_inches(12, 8)
        plt.savefig(os.path.join(self.log_loc, "progress.png"))
        plt.close(fig)


    def log(self, r, q_loss, q_var, p_norm, kl, epsilon):
        """ Log new metrics.

        Args:
            r (float): Latest reward
            q_loss (float): Latest Q loss
        """

        # save metrics
        self.returns.append(r)
        self.q_losses.append(q_loss)
        self.q_vars.append(q_var)
        self.p_norms.append(p_norm)
        self.kls.append(kl)
        self.epsilons.append(epsilon)

        # log metrics
        self.write()
        self.plot()


    def save(self, q_model, pi_model):
        """ Save model state dicts to folder.

        Args:
            q_model (torch.module): Q network
            pi_model (torch.module): Parameter network
        """
        torch.save(q_model.state_dict(), os.path.join(self.log_loc, "q_model.pt"))
        torch.save(pi_model.state_dict(), os.path.join(self.log_loc, "pi_model.pt"))