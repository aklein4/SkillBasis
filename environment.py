
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

import utils

class TrainingEnvironment:
    def __init__(self, model, env_name):
        self.env = gym.make(env_name)

        self.model = model
        
        self.baseline = 0
        

        
    def sampleEpisode(self):

        t = 0
        obs = utils.np2py(self.env.reset())
        seq = []

        states = []
        actions = []
        rewards = []

        while True:

            seq.append(obs)
            pi = self.model(torch.cat(obs))
            action = pi.sample()

            obs, reward, done, info = self.env.step(action)



            if done:
                break