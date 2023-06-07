
import numpy as np

import tkinter as tk
import time


CANVAS_SIZE = 500

DELTA_T = 0.1
FORCE = 1

ANG_CLIP = np.pi * 2
SPEED_CLIP = 5.0

BOUND = 20.0

DISCRETE_ACTIONS = np.array([
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 0],
    [0, 1],
    [1, -1],
    [1, 0],
    [1, 1]
])


class Drone:
    def __init__(self, discrete=False, render=False, max_t = 10):
        self.discrete = discrete

        self.t = None
        self.max_t = max_t

        self.pos = None
        self.vel = None
        
        self.ang = None
        self.ang_vel = None

        self.unit = CANVAS_SIZE / (BOUND * 2)
        self.canvas = None
        self.arrow = None
        self.circle = None

        self.rendering = render
        if self.rendering:
            self.root = tk.Tk()
            self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE)
            self.canvas.pack()
            
            self.arrow = self.canvas.create_line(0, 0, 0, 0, arrow=tk.LAST)
            self.circle = self.canvas.create_oval(0, 0, 0, 0)

        self.reset()


    def render(self):
        if self.rendering:

            # self.canvas.delete(self.arrow)
            self.canvas.coords(
                self.arrow,
                int(CANVAS_SIZE / 2 + self.unit*self.pos[0]),
                int(CANVAS_SIZE / 2 + self.unit*self.pos[1]),
                int(CANVAS_SIZE / 2 + self.unit*self.pos[0] + self.unit*np.cos(self.ang)), 
                int(CANVAS_SIZE / 2 + self.unit*self.pos[1] + self.unit*np.sin(self.ang)),
            )
            
            self.canvas.coords(
                self.circle,
                int(CANVAS_SIZE / 2 + self.unit*self.pos[0] - self.unit),
                int(CANVAS_SIZE / 2 + self.unit*self.pos[1] - self.unit),
                int(CANVAS_SIZE / 2 + self.unit*self.pos[0] + self.unit),
                int(CANVAS_SIZE / 2 + self.unit*self.pos[1] + self.unit)
            )

            self.root.update()

    
    def reset(self):
        self.t = 0

        self.pos = np.zeros(2)
        self.speed = np.zeros(1)

        self.ang = np.zeros(1)
        self.ang_vel = np.zeros(1)

        return self.getState()
    

    def getState(self):
        dir = np.array([np.cos(self.ang), np.sin(self.ang)])[:, 0]
        return np.concatenate([
            self.pos, dir, self.speed, self.ang_vel
        ], axis=0)
    

    def step(self, action):
        if self.discrete:
            action = DISCRETE_ACTIONS[action]

        # apply acceleration        
        self.ang_vel += (action[0] - action[1]) * DELTA_T * FORCE
        self.speed += (action[0] + action[1]) * DELTA_T * FORCE

        # clip speed
        self.ang_vel = np.clip(self.ang_vel, -ANG_CLIP, ANG_CLIP)
        self.speed = np.clip(self.speed, -SPEED_CLIP, SPEED_CLIP)

        # apply velocity
        self.ang += self.ang_vel * DELTA_T
        self.pos[0] += np.cos(self.ang) * self.speed * DELTA_T
        self.pos[1] += np.sin(self.ang) * self.speed * DELTA_T

        self.t += DELTA_T

        # check done
        if np.max(np.abs(self.pos)) > BOUND or self.t >= self.max_t:
            return self.reset(), 0, True, None

        return self.getState(), 0, False, None


def main():
    drone = Drone(discrete=True, render=True)
    drone.reset()

    while True:
        drone.step(5)
        drone.render()
        time.sleep(DELTA_T)


if __name__ == "__main__":
    main()