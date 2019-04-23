# simple grid environment that wraps around

import numpy as np

class GridEnv():
    def __init__(self, size):
        self.size = size
        self.state = None
        self.num_actions = 4    # left, right, up, down
        self.np_random = np.RandomState()

    def reset(self):
        self.state = np.array([0, 0])

    def sample(self):
        return self.np_random.choice(self.action_space)

    def step(self, action):
        assert action in self.action_space
        assert self.state is not None

        done = False
        reward = 0
        if action == 0:     #left
            self.state[0] = (self.state[0] - 1)%self.size
        elif action == 1:   #right
            self.state[0] = (self.state[0] + 1)%self.size
        elif action == 2:   #down
            self.state[1] = (self.state[1] + 1)%self.size
        else: #if action == 3:   #down
            self.state[1] = (self.state[1] - 1)%self.size
        
        if self.state[0] == self.size//2 and self.state[1] == self.size//2:
            done = True
            reward = 1
        return self.state, reward, done, {}
