
import torch

import os
import csv
import matplotlib.pyplot as plt


class Logger:
    def __init__(self,
            pi_model,
            encoder_model,
            basis_model,
            baseline_model,
            log_loc,
            save_every
        ):

        self.pi_model = pi_model
        self.encoder_model = encoder_model
        self.basis_model = basis_model
        self.baseline_model = baseline_model

        # metrics to track
        self.losses = []
        self.baseline_losses = []
        self.sigmas = []
        self.norms = []

        # save location
        self.log_loc = log_loc
        self.save_every = save_every

        # make folder
        os.makedirs(self.log_loc, exist_ok=True)
        
        # initialize metric csv
        with open(os.path.join(self.log_loc, "metrics.csv"), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow(["epoch"]  + ["loss", "baseline_loss", "sigma", "norm"])


    def write(self):
        with open(os.path.join(self.log_loc, "metrics.csv"), 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.losses)-1] + [self.losses[-1], self.baseline_losses[-1], self.sigmas[-1]])


    def plot(self):

        fig, ax = plt.subplots(2, 2)
        
        ax[0, 0].plot(self.losses)
        ax[0, 1].plot(self.baseline_losses)

        ax[1, 0].plot(self.sigmas)
        ax[1, 1].plot(self.norms)

        fig.tight_layout()
        fig.set_size_inches(10, 6)
        plt.savefig(os.path.join(self.log_loc, "progress.png"))
        plt.close(fig)


    def log(self, loss, baseline_loss):

        # save metrics
        self.losses.append(loss)
        self.baseline_losses.append(baseline_loss)

        self.sigmas.append(torch.mean(torch.exp(self.basis_model.log_sigma)).item())
        self.norms.append(torch.mean(torch.norm(self.basis_model.basis, dim=-1)).item())

        # log metrics
        self.write()
        self.plot()

        if len(self.losses)-1 % self.save_every == 0:
            self.save()


    def save(self):
        torch.save(self.pi_model.state_dict(), os.path.join(self.log_loc, "pi_model.pt"))
        torch.save(self.encoder_model.state_dict(), os.path.join(self.log_loc, "encoder_model.pt"))
        torch.save(self.basis_model.state_dict(), os.path.join(self.log_loc, "basis_model.pt"))
        torch.save(self.baseline_model.state_dict(), os.path.join(self.log_loc, "baseline_model.pt"))