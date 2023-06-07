


import torch
import numpy as np

import matplotlib.pyplot as plt


_ACTIONS = np.array([
    [0, 1],
    [0, -1],
    [1, 0],
    [-1, 0]
])

STATIC = True

FOOD_REWARD = 1
BAD_PENALTY = -0.5


class GridWorld:
    def __init__(self, board_size, num_modes, num_food, time_limit):

        self.board_size = board_size
        self.num_modes = num_modes
        self.num_food = num_food
        self.time_limit = time_limit

        self.board = None

        self.t = None
        self.char_ind = None
        self.food_left = None

        self.target_food = None


    def reset(self, target_food):
        self.t = 0
        self.target_food = target_food
        self.food_left = self.num_food * len(target_food)

        # initialize the board with all empty
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        avail_set = set([(i, j) for i in range(self.board_size) for j in range(self.board_size)])
        
        # character starts at 0, 0
        self.char_ind = np.array([0, 0])
        avail_set.remove((0, 0))
        self.board[0, 0] = 1

        # add the food
        if STATIC:
            np.random.seed(0)
        for i in range(2, self.num_modes+2):
            for _ in range(self.num_food):
                if len(avail_set) == 0:
                    raise RuntimeError("Not enough space for food")
                
                ind = list(avail_set)[np.random.choice(len(avail_set))]
                self.board[ind[0], ind[1]] = i
                avail_set.remove(ind)

        return self.getState()
    

    def _getNewPos(self, action):
        uncut = self.char_ind + _ACTIONS[action]
        uncut += self.board_size
        
        return uncut % self.board_size


    def step(self, action):
        assert action >= 0
        assert action < 4

        reward = 0

        # take action
        new_ind = self._getNewPos(action)
        
        if self.board[new_ind[0], new_ind[1]]-3 in self.target_food:
            self.food_left -= 1
            reward = FOOD_REWARD

        elif self.board[new_ind[0], new_ind[1]] > 1:
            reward = BAD_PENALTY

        self.board[self.char_ind[0], self.char_ind[1]] = 0
        self.board[new_ind[0], new_ind[1]] = 1

        self.char_ind = new_ind

        # check done
        self.t += 1
        done = self.t >= self.time_limit or self.food_left == 0

        return self.getState(), reward, done, None
    

    def getState(self):
        return self.board


    def render(self):
        print("\n", self.board, "\n")


def main():
    env = GridWorld(10, 3, 5, 100)
    env.reset([1])
    while True:
        env.render()
        action = int(input("action: "))
        obs, reward, done, info = env.step(action)
        if done:
            break


if __name__ == "__main__":
    main()