
import numpy as np

import tkinter as tk
import time


RANDOM_RESET = False

CANVAS_SIZE = 500

DELTA_T = 0.1
FORCE = 5
TORQUE = np.pi * 2

MAX_SPEED = 5.0

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
    def __init__(self, discrete=False, render=False, max_t = 10, boxes=None, target=np.array([19, 19])):
        self.discrete = discrete
        self.boxes = boxes
        self.target = target

        self.t = None
        self.max_t = max_t

        self.pos = None
        self.vel = None
        
        self.ang = None

        self.unit = CANVAS_SIZE / (BOUND * 2)
        self.canvas = None
        self.arrow = None
        self.circle = None

        self.rendering = render
        if self.rendering:
            print("Rendering...")
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
            self.speed = np.zeros(1)
            self.ang = np.random.uniform(-np.pi, np.pi, 1)

        else:
            self.pos = np.array([0, 0])
            self.speed = np.zeros(1)
            self.ang = np.zeros(1)

        return self.getState()
    

    def getState(self):
        dir = np.array([np.cos(self.ang), np.sin(self.ang)])[:, 0]
        state = np.concatenate([
            self.pos / BOUND, dir, self.speed / MAX_SPEED
        ], axis=0)
    
        return state
    
    
    def checkCollision(self, p):
        if self.boxes is None:
            return False   
        
        x, y = p[0], p[1]
        
        for i in self.boxes.shape[0]:
            b = self.boxes[i]
            if x >= b[0] and x <= b[1] and y >= b[2] and y <= b[3]:
                return True
            
        return False
    

    def step(self, action):
        if self.discrete:
            action = DISCRETE_ACTIONS[action]

        # apply acceleration        
        self.speed += action[0] * DELTA_T * FORCE

        # clip speed
        self.speed = np.clip(self.speed, -MAX_SPEED, MAX_SPEED)

        # apply velocity
        self.ang += action[1] * DELTA_T * TORQUE
        old_pos = self.pos.copy()
        self.pos = self.pos + np.array([
            np.cos(self.ang) * self.speed * DELTA_T,
            np.sin(self.ang) * self.speed * DELTA_T
        ]).squeeze(-1)

        # clip position
        self.pos = np.clip(self.pos, -BOUND+0.01, BOUND-0.01)

        # apply walls
        if self.checkCollision(self.pos):
            return self.reset(), 0, True, None     

        # check done
        self.t += DELTA_T
        if self.t >= self.max_t:
            return self.reset(), 0, True, None

        # get reward for moving closer
        reward = -(np.linalg.norm(self.pos - self.target) - np.linalg.norm(old_pos - self.target))

        # check if at target
        if np.linalg.norm(self.pos - self.target) < 1:
            return self.reset(), reward, True, None

        # regular step
        return self.getState(), reward, False, None


def main():
    drone = Drone(discrete=True, render=True)
    drone.reset()

    while True:
        drone.step(np.random.randint(0, 9))
        drone.render()
        time.sleep(DELTA_T)


if __name__ == "__main__":
    main()