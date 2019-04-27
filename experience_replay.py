import numpy as np

# from OpenAI's baselines https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class ExperienceReplay():
    def __init__(self, size):
        self.buffer = []
        self.maxsize = size
        self.next_idx = 0
        self.np_random = np.random.RandomState()

    def __len__(self):
        return len(self.buffer)
    
    def add(self, experience):
        if self.next_idx >= len(self.buffer):   # increase size of buffer if there's still room
            self.buffer.append(experience)
        else:                                   # overwrite old experience
            self.buffer[self.next_idx] = experience
        self.next_idx = (self.next_idx + 1)%self.maxsize

    def sample(self, batch_size):
        # sample indices into buffer
        idxs = self.np_random.randint(0,len(self.buffer),size=(batch_size,))    # randint samples ints from [low,high)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in idxs:    # extract experience at given indices
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)    # list of int arrays
            actions.append(action)  # list of ints
            rewards.append(reward)  # list of ints
            next_states.append(next_state)  # list of int arrays
            dones.append(done)  # list of bools
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))
