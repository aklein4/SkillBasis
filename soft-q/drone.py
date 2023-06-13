
import numpy as np

import tkinter as tk
import time


RANDOM_RESET = False

CANVAS_SIZE = 500

DELTA_T = 0.05
FORCE = 8
TORQUE = np.pi

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
    def __init__(self, discrete=False, render=False, max_t = 3, boxes=None, target=np.array([19, 19])):
        self.discrete = discrete
        self.boxes = boxes
        self.target = target

        self.batch_size = None
        self.t = None
        self.max_t = max_t

        self.pos = None
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
                int(CANVAS_SIZE / 2 + self.unit*self.pos[0,0]),
                int(CANVAS_SIZE / 2 + self.unit*self.pos[0,1]),
                int(CANVAS_SIZE / 2 + self.unit*self.pos[0,0] + self.unit*np.cos(self.ang[0])), 
                int(CANVAS_SIZE / 2 + self.unit*self.pos[0,1] + self.unit*np.sin(self.ang[0])),
            )
            
            self.canvas.coords(
                self.circle,
                int(CANVAS_SIZE / 2 + self.unit*self.pos[0,0] - self.unit),
                int(CANVAS_SIZE / 2 + self.unit*self.pos[0,1] - self.unit),
                int(CANVAS_SIZE / 2 + self.unit*self.pos[0,0] + self.unit),
                int(CANVAS_SIZE / 2 + self.unit*self.pos[0,1] + self.unit)
            )

            self.root.update()

            time.sleep(DELTA_T)

    
    def reset(self, batch_size=1):
        self.t = 0
        self.batch_size = batch_size

        if RANDOM_RESET:
            self.pos = np.random.uniform(-BOUND, BOUND, (batch_size, 2))
            self.ang = np.random.uniform(-np.pi, np.pi, (batch_size, 1))

        else:
            self.pos = np.zeros((batch_size, 2))
            self.ang = np.zeros((batch_size, 1))

        return self.getState()
    

    def getState(self):
        dir = np.concatenate([np.cos(self.ang), np.sin(self.ang)], axis=-1)
        state = np.concatenate([
            self.pos / BOUND, dir
        ], axis=-1)

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

        # apply velocity
        self.ang += action[:,1:] * DELTA_T * TORQUE
        old_pos = self.pos.copy()
        self.pos[:,:1] += np.cos(self.ang) * action[:,:1] * FORCE * DELTA_T
        self.pos[:,1:] += np.sin(self.ang) * action[:,:1] * FORCE * DELTA_T

        # clip position
        self.pos = np.clip(self.pos, -BOUND+0.01, BOUND-0.01)

        # apply walls
        if self.checkCollision(self.pos):
            return None, 0, True, None     

        # check done
        self.t += DELTA_T
        if self.t >= self.max_t:
            return self.getState(), 0, True, None

        return self.getState(), 0, False, None

        # # get reward for moving closer
        # reward = -(np.linalg.norm(self.pos - self.target) - np.linalg.norm(old_pos - self.target))

        # # check if at target
        # if np.linalg.norm(self.pos - self.target) < 1:
        #     return self.reset(), reward, True, None

        # # regular step
        # return self.getState(), reward, False, None


def main():
    drone = Drone(discrete=True, render=True)
    drone.reset()

    while True:
        drone.step(np.random.randint(0, 9))
        drone.render()
        time.sleep(DELTA_T)


if __name__ == "__main__":
    main()