
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
        self.kls = []
        self.switch_percs = []
        self.skill_lens = []

        # save location
        self.log_loc = log_loc

        # make folder
        os.makedirs(self.log_loc, exist_ok=True)
        
        # initialize metric csv
        with open(os.path.join(self.log_loc, "metrics.csv"), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow(["epoch", "return", "kl", "switch_perc", "skill_len"])

        # initialize gene csv
        with open(os.path.join(self.log_loc, "genes.csv"), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow(["gene_sequence"])


    def write(self):
        """
        Write latest metrics to csv.
        """
        with open(os.path.join(self.log_loc, "metrics.csv"), 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.returns)-1, self.returns[-1], self.kls[-1], self.switch_percs[-1], self.skill_lens[-1]])


    def print_genes(self, genes):
        with open(os.path.join(self.log_loc, "genes.csv"), 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow(genes)


    def plot(self):
        """
        Plot the metrics' progress.
        """

        # init figure
        fig, ax = plt.subplots(2, 2)

        # plot returns
        ax[0, 0].plot(self.returns)
        ax[0, 0].set_title("Average Returns (Smoothed)")
        ax[0, 0].set_ylabel('avg total reward')

        # plot p_rollings
        ax[0, 1].plot(self.kls)
        ax[0, 1].set_title("Inter-Gene Policy KL Divergences")
        ax[0, 1].set_ylabel('avg kl')

        # plot switch_percs
        ax[1, 0].plot(self.switch_percs)
        ax[1, 0].set_title("Gene Switching Percentage")
        ax[1, 0].set_ylabel('% of steps')

        # plot skill_lens
        ax[1, 1].plot(self.skill_lens)
        ax[1, 1].set_title("Average Skill Length")
        ax[1, 1].set_ylabel('avg length in steps')

        # save figure to file
        fig.suptitle("Training Progress (Smoothed)")
        fig.tight_layout()
        # fig.set_size_inches(8, 9)
        plt.savefig(os.path.join(self.log_loc, "progress.png"))
        plt.close(fig)


    def log(self, r, kl, switch_perc, skill_len, genes):
        """ Log new metrics.

        Args:
            r (float): Latest reward
            q_loss (float): Latest Q loss
        """

        # save metrics
        self.returns.append(r)
        self.kls.append(kl)
        self.switch_percs.append(switch_perc)
        self.skill_lens.append(skill_len)

        # log metrics
        self.write()
        self.plot()
        self.print_genes(genes)


    def save(self, epi_model, pi_model):
        """ Save model state dicts to folder.

        Args:
            q_model (torch.module): Q network
            pi_model (torch.module): Parameter network
        """
        torch.save(epi_model.state_dict(), os.path.join(self.log_loc, "epi_model.pt"))
        torch.save(pi_model.state_dict(), os.path.join(self.log_loc, "pi_model.pt"))