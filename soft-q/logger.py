
import torch

import utils

import os
import csv
import matplotlib.pyplot as plt
import numpy as np


class Logger:
    def __init__(self,
            pi_model,
            encoder_model,
            log_loc,
            save_every
        ):

        self.pi_model = pi_model
        self.encoder_model = encoder_model

        # metrics to track
        self.informations = []
        self.q_losses = []
        self.norms = []
        self.entropies = []

        # save location
        self.log_loc = log_loc
        self.save_every = save_every

        # create a grid for vizualization
        x_coords = np.linspace(-1, 1, 20)
        y_coords = np.linspace(-1, 1, 20)
        grid = np.stack(np.meshgrid(x_coords, y_coords))
        grid = torch.from_numpy(grid).float().to(utils.DEVICE).permute(1, 2, 0)
        self.grid = torch.zeros(20, 20, 4).to(utils.DEVICE)
        self.grid[:, :, :2] = grid

        # make folder
        os.makedirs(self.log_loc, exist_ok=True)
        
        # initialize metric csv
        with open(os.path.join(self.log_loc, "metrics.csv"), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow(["epoch"]  + ["information", "q_loss", "norm", "entropy"])


    def write(self):
        with open(os.path.join(self.log_loc, "metrics.csv"), 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.informations)-1] + [self.informations[-1], self.q_losses[-1], self.norms[-1], self.entropies[-1]])


    def plot(self):

        fig, ax = plt.subplots(2, 2)
        
        ax[0, 0].plot(self.informations)
        ax[0, 0].set_title("Skill-Transition Mutual Information")
        
        ax[0, 1].plot(self.q_losses)
        ax[0, 1].set_title("Q-value MSE Loss")

        activations = torch.zeros(20, 20, 3).to(utils.DEVICE)
        activations[:, :, :2] = self.encoder_model(self.grid[:, :, :2])
        activations[:, :, 2] = activations[:, :, 1]
        activations[:, :, 1] = activations[:, :, 0]
        for i in range(3):
            activations[:, :, i] -= torch.min(activations[:, :, i])
            activations[:, :, i] /= torch.max(activations[:, :, i])
        activations = torch.sigmoid(activations)

        ax[1, 0].imshow(utils.torch2np(activations))
        ax[1, 0].set_title("Latent Space Representation")
        ax[1, 0].set_xlabel("Iteration (8 episodes per)")
        
        ax[1, 1].plot(self.entropies)
        ax[1, 1].set_title("Policy Entropy")

        fig.tight_layout()
        fig.set_size_inches(10, 7)
        plt.savefig(os.path.join(self.log_loc, "progress.png"))
        plt.close(fig)


    def log(self, information, q_loss, norm, entrop):

        # save metrics
        self.informations.append(information)
        self.q_losses.append(q_loss)
        self.norms.append(norm)
        self.entropies.append(entrop)

        self.write()

        if (len(self.informations)-1) % self.save_every == 0:
            self.save()


    def save(self):
        self.plot()
        torch.save(self.pi_model.state_dict(), os.path.join(self.log_loc, "pi_model.pt"))
        torch.save(self.encoder_model.state_dict(), os.path.join(self.log_loc, "encoder_model.pt"))