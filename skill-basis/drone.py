
import numpy as np

import tkinter as tk
import time


RANDOM_RESET = True

CANVAS_SIZE = 500

DELTA_T = 0.1
FORCE = 5
DRAG = 0.25

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

            time.sleep(DELTA_T)

    
    def reset(self):
        self.t = 0

        if RANDOM_RESET:
            self.pos = np.random.uniform(-BOUND, BOUND, 2)
            self.speed = np.random.normal(0, FORCE/2, 1)
            self.ang = np.random.uniform(-np.pi, np.pi, 1)
            self.ang_vel = np.random.normal(0, np.pi/4, 1)

        else:
            self.pos = np.zeros(2)
            self.speed = np.zeros(1)
            self.ang = np.zeros(1)
            self.ang_vel = np.zeros(1)

        return self.getState()
    

    def getState(self):
        dir = np.array([np.cos(self.ang), np.sin(self.ang)])[:, 0]
        state = np.concatenate([
            self.pos, dir, self.speed, self.ang_vel
        ], axis=0)

        return state
    

    def step(self, action):
        if self.discrete:
            action = DISCRETE_ACTIONS[action]

        # apply drag
        action[0] -= DRAG * (self.speed + self.ang_vel)
        action[1] -= DRAG * (self.speed - self.ang_vel)

        # apply acceleration        
        self.ang_vel += (action[0] - action[1]) * DELTA_T * FORCE
        self.speed += (action[0] + action[1]) * DELTA_T * FORCE

        # apply velocity
        self.ang += self.ang_vel * DELTA_T
        self.pos[0] += np.cos(self.ang) * self.speed * DELTA_T
        self.pos[1] += np.sin(self.ang) * self.speed * DELTA_T

        # apply walls
        self.pos = np.clip(self.pos, -BOUND, BOUND)        

        # check done
        self.t += DELTA_T
        if self.t >= self.max_t:
            return self.reset(), 0, True, None

        return self.getState(), 0, False, None


def main():
    drone = Drone(discrete=True, render=True)
    drone.reset()

    while True:
        drone.step(np.random.randint(0, 9))
        drone.render()
        time.sleep(DELTA_T)


if __name__ == "__main__":
    main()