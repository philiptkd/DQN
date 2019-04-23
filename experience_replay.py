import numpy as np

class ExperienceReplay():
    def __init__(self, size):
        self.buffer = []
        self.maxsize = size
        self.next_idx = 0
        self.np_random = np.RandomState()

    def __len__(self):
        return len(self.buffer)
    
    def add(self, experience):
        if self.next_idx >= len(self.buffer):   # increase size of buffer if there's still room
            self.buffer.append(experience)
        else:                                   # overwrite old experience
            self.buffer[self.next_idx] = experience
        self.next_idx = (self.next_idx + 1)%self.maxsize

    def sample(self, batch_size):
        idxs = self.np_random.randint(0,len(self.buffer),size=(batch_size,))    # randint samples ints from [low,high)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in idxs:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(np.array(state))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state))
            dones.append(done)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done))
