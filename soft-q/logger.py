
import torch

import os
import csv
import matplotlib.pyplot as plt


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

        ax[1, 0].plot(self.norms)
        ax[1, 0].set_title("delta_l L1 Norm")
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