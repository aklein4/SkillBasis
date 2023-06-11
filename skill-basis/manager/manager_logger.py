
import torch

import os
import csv
import matplotlib.pyplot as plt


class Logger:
    def __init__(self,
            man_model,
            baseline_model,
            log_loc,
            save_every
        ):

        self.man_model = man_model
        self.baseline_model = baseline_model

        # metrics to track
        self.losses = []
        self.bases = []

        # save location
        self.log_loc = log_loc
        self.save_every = save_every

        # make folder
        os.makedirs(self.log_loc, exist_ok=True)
        
        # initialize metric csv
        with open(os.path.join(self.log_loc, "metrics.csv"), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow(["epoch"]  + ["rewards", "bases"])


    def write(self):
        with open(os.path.join(self.log_loc, "metrics.csv"), 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.losses)-1] + [self.losses[-1], self.bases[-1]])


    def plot(self):

        plt.plot(self.losses)
        plt.title("Rewards Through Manager Training")
        plt.xlabel("Iteration")
        plt.savefig(os.path.join(self.log_loc, "progress.png"))
        plt.clf()


    def log(self, loss, base):

        # save metrics
        self.losses.append(loss)
        self.bases.append(base)

        self.write()

        if (len(self.losses)-1) % self.save_every == 0:
            self.save()


    def save(self):
        self.plot()
        torch.save(self.baseline_model.state_dict(), os.path.join(self.log_loc, "baseline_model.pt"))
        torch.save(self.man_model.state_dict(), os.path.join(self.log_loc, "man_model.pt"))