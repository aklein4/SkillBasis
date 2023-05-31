
import torch
import numpy as np

import matplotlib.pyplot as plt

EMPTY = 0
WALL = 1
GHOST = 2
PACMAN = 3

BOARD = """
m______#______m
_#####_#_#####_
_______________
_####_###_####_
____#__#__#____
###_##_#_##_###
#___#__@__#___#
#_#___###___#_#
#___#_____#___#
###_##_#_##_###
____#__#__#____
_####_###_####_
_______________
_#####_#_#####_
m______#______m
"""
BOARD = BOARD[1:]

BOARD_WIDTH = BOARD.find("\n")
BOARD_HEIGHT = BOARD.count("\n")

BOARD = BOARD.replace("_", str(EMPTY))
BOARD = BOARD.replace("#", str(WALL))
BOARD = BOARD.replace("m", str(GHOST))
BOARD = BOARD.replace("@", str(PACMAN))
BOARD = BOARD.replace("\n", "")


FOOD_REWARD = 1
DEATH_REWARD = -100

GHOST_DELAY = 3


class Pacman:
    def __init__(self):

        self.board = None
        self.food = None

        self.num_ghosts = None
        self.ghost_inds = None
        self.ghost_vels = None

        self.pacman_ind = None

        self.t = None
        self.reset()


    def reset(self):
        self.board = np.array([int(i) for i in BOARD]).reshape((BOARD_HEIGHT, BOARD_WIDTH))
        self.food = (self.board != WALL).astype(int)

        self.num_ghosts = np.sum(self.board == GHOST)
        self.ghost_inds =  np.stack((self.board == GHOST).nonzero()).T
        self.ghost_vels = np.zeros((self.num_ghosts, 2), dtype=int)
        self.ghost_vels[:, 0] = 1

        self.pacman_ind = np.concatenate((self.board == PACMAN).nonzero())

        self.t = 0
        return self.getState(), None


    def _checkWall(self, loc):
        if loc[0] < 0 or loc[0] >= BOARD_HEIGHT:
            return True
        if loc[1] < 0 or loc[1] >= BOARD_WIDTH:
            return True
        if self.board[loc[0], loc[1]] == WALL:
            return True
        return False


    def _seekPacman(self, loc):
        if loc[0] == self.pacman_ind[0]:
            if loc[1] < self.pacman_ind[1]:
                for i in range(loc[1], self.pacman_ind[1]):
                    if self.board[loc[0], i] == WALL:
                        return None
                return np.array([0, 1])
            else:
                for i in range(self.pacman_ind[1], loc[1]):
                    if self.board[loc[0], i] == WALL:
                        return None
                return np.array([0, -1])
        
        if loc[1] == self.pacman_ind[1]:
            if loc[0] < self.pacman_ind[0]:
                for i in range(loc[0], self.pacman_ind[0]):
                    if self.board[i, loc[1]] == WALL:
                        return None
                return np.array([1, 0])
            else:
                for i in range(self.pacman_ind[0], loc[0]):
                    if self.board[i, loc[1]] == WALL:
                        return None
                return np.array([-1, 0])

        return None


    def _getGhostVel(self, loc, curr_vel):
        seek = self._seekPacman(loc)
        if seek is not None:
            return seek
        
        checkers = np.array([
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0]
        ])
        keep = [True] * 4
        for i in range(4):
            if np.sum(np.abs((curr_vel + checkers[i]))) == 0:
                keep[i] = False
        checkers = checkers[keep]

        avail = [not(self._checkWall(loc + checkers[i])) for i in range(len(checkers))]
        checkers = checkers[avail]

        return checkers[np.random.choice(checkers.shape[0])]


    def getState(self):
        return np.stack([self.board, self.food], axis=0)
    

    def step(self, action):
        reward = 0
        self.t += 1

        # move pacman
        actions = np.array([
            [0, 0],
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0]
        ])
        new_pacman_ind = self.pacman_ind + actions[action]
        if not self._checkWall(new_pacman_ind):
            self.board[self.pacman_ind[0], self.pacman_ind[1]] = EMPTY
            self.pacman_ind = new_pacman_ind
            self.board[self.pacman_ind[0], self.pacman_ind[1]] = PACMAN
            if self.food[self.pacman_ind[0], self.pacman_ind[1]] == 1:
                self.food[self.pacman_ind[0], self.pacman_ind[1]] = 0
                reward += FOOD_REWARD

        # check ghost collision
        if np.any(np.array([np.all(self.ghost_inds[i] == self.pacman_ind) for i in range(self.num_ghosts)])):
            return self.reset()[0], reward + DEATH_REWARD, True, None, None

        # delay -> do nothing
        if self.t % GHOST_DELAY == 0:
            return self.getState(), reward, False, None, None

        # move ghosts
        for i in range(self.num_ghosts):
            self.ghost_vels[i] = self._getGhostVel(self.ghost_inds[i], self.ghost_vels[i])
            new_pos = self.ghost_inds[i] + self.ghost_vels[i]
            if self.board[new_pos[0], new_pos[1]] == GHOST:
                continue
            self.board[self.ghost_inds[i][0], self.ghost_inds[i][1]] = EMPTY
            self.ghost_inds[i] = new_pos
            self.board[self.ghost_inds[i][0], self.ghost_inds[i][1]] = GHOST

        # check ghost collision
        if np.any(np.array([np.all(self.ghost_inds[i] == self.pacman_ind) for i in range(self.num_ghosts)])):
            return self.reset()[0], reward + DEATH_REWARD, True, None, None

        return self.getState(), reward, False, None, None
    

    def render(self):
        print("\n", self.board, "\n")


def main():
    env = Pacman()
    obs = env.reset()
    while True:
        env.render()
        action = np.random.randint(5)
        obs, reward, done, info = env.step(action)
        if done:
            break


if __name__ == "__main__":
    main()