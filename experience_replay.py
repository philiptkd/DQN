import numpy as np
from segment_tree import MinSegmentTree, SumSegmentTree
from binary_heap import Heap, HeapItem

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
        return self._encode_samples(idxs)
        
    def _encode_samples(self, idxs):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in idxs:    # extract experience at given indices
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)    # list of int arrays
            actions.append(action)  # list of ints
            rewards.append(reward)  # list of ints
            next_states.append(next_state)  # list of int arrays
            dones.append(done)  # list of bools
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

# proportional sampling as implemented by OpenAI
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class ProportionalReplay(ExperienceReplay):
    def __init__(self, size, alpha):
        super(ProportionalReplay, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        self._tree_size = 1
        while self._tree_size < self.maxsize:
            self._tree_size *= 2

        self._min_tree = MinSegmentTree(self._tree_size)    # for calculating maximum IS weight
        self._sum_tree = SumSegmentTree(self._tree_size)    # for proportional sampling
        self._max_priority = 1.0    # will change as priorities are updated according to TD error

    def add(self, experience):
        idx = self.next_idx     # save idx before it's changed in super call
        super().add(experience) # put experience data (s,a,r,s',done) in buffer

        # give new experience max priority to ensure it's replayed at least once
        self._min_tree[idx] = self._max_priority ** self._alpha 
        self._sum_tree[idx] = self._max_priority ** self._alpha

    # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges. 
    # Next, a value is uniformly sampled from each range.
    def _sample_proportional(self, batch_size):
        idxs = []
        p_total = self._sum_tree.sum(0, len(self.buffer)-1) # sum of the priorities of all experience in the buffer
        every_range_len = p_total / batch_size  # length of every range over [0,p_total] (batch_size = k)
        for i in range(batch_size): # for each range
            mass = self.np_random.uniform()*every_range_len + i*every_range_len  # uniformly sampling a probability mass from this range
            idx = self._sum_tree.find_prefixsum_idx(mass) # get smallest experience index s.t. cumulative dist F(idx) >= mass
            idxs.append(idx)
        return idxs

    # sample batch of experiences along with their weights and indices
    def sample(self, batch_size, beta):
        assert beta > 0
        idxs = self._sample_proportional(batch_size)    # sampled experience indices

        weights = []
        p_min = self._min_tree.min() / self._sum_tree.sum() # minimum possible priority for a transition
        max_weight = (p_min * len(self.buffer)) ** (-beta)    # (p_uniform/p_min)^beta is maximum possible IS weight

        # get IS weights for sampled experience
        for idx in idxs:
            p_sample = self._sum_tree[idx] / self._sum_tree.sum()   # normalize sampled priority
            weight = (p_sample * len(self.buffer)) ** (-beta) # (p_uniform/p_sample)^beta. IS weight
            weights.append(weight / max_weight) # weights normalized by max so that they only scale the update downwards
        weights = np.array(weights)

        encoded_sample = self._encode_samples(idxs) # collect experience at given indices 
        return tuple(list(encoded_sample) + [weights, idxs])

    # set the priorities of experiences at given indices 
    def update_priorities(self, idxs, priorities):
        assert len(idxs) == len(priorities)
        for idx, priority in zip(idxs, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)
            self._sum_tree[idx] = priority ** self._alpha
            self._min_tree[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

# TODO: write versions of every method in ProportionalReplay
class RankBasedReplay(ExperienceReplay):
    def __init__(self, size, alpha):
        super(RankBasedReplay, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        min_tree_size = 1
        while min_tree_size < self.maxsize:
            min_tree_size *= 2

        self._min_tree = MinSegmentTree(min_tree_size)    # for calculating maximum IS weight
        self._max_heap = Heap(self.maxsize)
        self._max_priority = 1.0    # will change as priorities are updated according to TD error


