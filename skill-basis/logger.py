
import torch

import os
import csv
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, pi, baseline, n_modes, log_loc, save_every):
        """ Logger to save training metrics.

        Args:
            log_loc (str): Folder for saving
        """

        self.pi = pi
        self.baseline = baseline

        self.n_modes = n_modes

        # metrics to track
        self.returns = [[] for _ in range(n_modes)]

        # save location
        self.log_loc = log_loc
        self.save_every = save_every

        # make folder
        os.makedirs(self.log_loc, exist_ok=True)
        
        # initialize metric csv
        with open(os.path.join(self.log_loc, "metrics.csv"), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow(["epoch"]  + ["return_{}".format(i) for i in range(n_modes)])


    def write(self):
        """
        Write latest metrics to csv.
        """
        with open(os.path.join(self.log_loc, "metrics.csv"), 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.returns)-1] + [self.returns[i][-1] for i in range(self.n_modes)])


    def plot(self):
        """
        Plot the metrics' progress.
        """

        # plot returns
        for i in range(self.n_modes):
            plt.plot(self.returns[i])
        plt.savefig(os.path.join(self.log_loc, "progress.png"))
        plt.clf()

        # save values
        # self.pi.net.showValues(os.path.join(self.log_loc, "genes.png"))


    def log(self, r):
        """ Log new metrics.

        Args:
            r (float): Latest reward
            q_loss (float): Latest Q loss
        """

        # save metrics
        for i in range(self.n_modes):
            self.returns[i].append(r[i])

        # log metrics
        self.write()
        self.plot()

        if len(self.returns)-1 % self.save_every == 0:
            self.save()


    def save(self):
        """ Save model state dicts to folder.

        Args:
            q_model (torch.module): Q network
            pi_model (torch.module): Parameter network
        """
        torch.save(self.baseline.state_dict(), os.path.join(self.log_loc, "baseline_model.pt"))
        torch.save(self.pi.state_dict(), os.path.join(self.log_loc, "pi_model.pt"))