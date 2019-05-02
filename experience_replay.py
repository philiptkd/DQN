import numpy as np
from segment_tree import MinSegmentTree, SumSegmentTree
from binary_heap import Heap, HeapItem
from zipf import load_quantiles

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
        return self.encode_samples(idxs)
        
    def encode_samples(self, idxs, ranked_priority=False):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in idxs:    # extract experience at given indices
            if ranked_priority:
                state, action, reward, next_state, done = self.buffer[idx][0]
            else:
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
        self.alpha = alpha

        self.tree_size = 1
        while self.tree_size < self.maxsize:
            self.tree_size *= 2

        self.min_tree = MinSegmentTree(self.tree_size)    # for calculating maximum IS weight
        self.sum_tree = SumSegmentTree(self.tree_size)    # for proportional sampling
        self.max_priority = 1.0   # maximum priority we've seen so far. will be updated

    def add(self, experience):
        idx = self.next_idx     # save idx before it's changed in super call
        super().add(experience) # put experience data (s,a,r,s',done) in buffer

        # give new experience max priority to ensure it's replayed at least once
        self.min_tree[idx] = self.max_priority ** self.alpha 
        self.sum_tree[idx] = self.max_priority ** self.alpha

    # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges. 
    # Next, a value is uniformly sampled from each range.
    def sample_proportional(self, batch_size):
        idxs = []
        p_total = self.sum_tree.sum(0, len(self.buffer)-1) # sum of the priorities of all experience in the buffer
        every_range_len = p_total / batch_size  # length of every range over [0,p_total] (batch_size = k)
        for i in range(batch_size): # for each range
            mass = self.np_random.uniform()*every_range_len + i*every_range_len  # uniformly sampling a probability mass from this range
            idx = self.sum_tree.find_prefixsum_idx(mass) # get smallest experience index s.t. cumulative dist F(idx) >= mass
            idxs.append(idx)
        return idxs

    # sample batch of experiences along with their weights and indices
    def sample(self, batch_size, beta):
        assert beta > 0
        idxs = self.sample_proportional(batch_size)    # sampled experience indices

        weights = []
        p_min = self.min_tree.min() / self.sum_tree.sum() # minimum possible priority for a transition
        max_weight = (p_min * len(self.buffer)) ** (-beta)    # (p_uniform/p_min)^beta is maximum possible IS weight

        # get IS weights for sampled experience
        for idx in idxs:
            p_sample = self.sum_tree[idx] / self.sum_tree.sum()   # normalize sampled priority
            weight = (p_sample * len(self.buffer)) ** (-beta) # (p_uniform/p_sample)^beta. IS weight
            weights.append(weight / max_weight) # weights normalized by max so that they only scale the update downwards
        weights = np.array(weights)

        encoded_sample = self.encode_samples(idxs) # collect experience at given indices 
        return tuple(list(encoded_sample) + [weights, idxs])

    # set the priorities of experiences at given indices 
    def update_priorities(self, idxs, priorities):
        assert len(idxs) == len(priorities)
        for idx, priority in zip(idxs, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

# assumes the caller has also called save_quantiles with the correct batch_size
class RankBasedReplay(ExperienceReplay):
    def __init__(self, size, alpha):
        super(RankBasedReplay, self).__init__(size)
        assert alpha >= 0
        self.alpha = alpha

        self.heap = Heap(self.maxsize)
        self.max_priority = 1.0    # will change as priorities are updated according to TD error
 
        self.N_list, self.range_list = load_quantiles()  # gets ranges of equal probability of zipf distribution for a few values of N
        self.range_idx = 0    # index into N_list of the ranges we're currently using
        self.priority_sums = [sum([i**(-alpha) for i in range(1,N+1)]) for N in self.N_list] # normalizing factors for priority distributions
        self.min_priorities = [N**(-alpha) / self.priority_sums[i] for i,N in enumerate(self.N_list)]   # minimum possible priorities given N

    def add(self, experience):
        if self.next_idx >= len(self.buffer):   # increase size of buffer if there's still room
            self.buffer.append([experience, self.next_idx]) # index is into the heap
            self.heap.insert(HeapItem(self.max_priority**self.alpha, self.next_idx)) # index is into buffer
        
        else:                                   # overwrite old experience
            self.buffer[self.next_idx][0] = experience
            heap_idx = self.buffer[self.next_idx][1]
            self.heap[heap_idx].value = self.max_priority**self.alpha

        self.next_idx = (self.next_idx + 1)%self.maxsize

        # update set of ranges we're using
        if self.range_idx < len(self.N_list)-1  and  len(self.buffer) >= self.N_list[self.range_idx + 1]:
            self.range_idx += 1

    # a rank is uniformly sampled from each of a set of precomputed ranges
    def sample_by_rank(self, batch_size):
        if len(self.buffer) < batch_size:       # return all indices if there are fewer than batch_size of them
            return list(range(1, len(self.buffer)+1)) 

        ranks = []
        ranges = self.range_list[self.range_idx]   # precomputed ranges
        for _range in ranges: # for each range
            ranks.append(self.np_random.randint(_range[0], _range[1]+1))   # random int in closed interval
        return ranks
    
    # sample batch of experiences along with their weights and indices
    def sample(self, batch_size, beta):
        assert beta > 0
        ranks = self.sample_by_rank(batch_size)

        p_min = self.min_priorities[self.range_idx] # minimum possible priority for a transition
        max_weight = (p_min * len(self.buffer)) ** (-beta)    # (p_uniform/p_min)^beta is maximum possible IS weight

        # get IS weights for sampled experience
        weights = []
        for rank in ranks:
            p_sample = rank**(-self.alpha)/self.priority_sums[self.range_idx]   # normalize sampled priority
            weight = (p_sample * len(self.buffer)) ** (-beta) # (p_uniform/p_sample)^beta. IS weight
            weights.append(weight / max_weight) # weights normalized by max so that they only scale the update downwards
        weights = np.array(weights)

        heap_idxs = [self.heap.get_kth_largest(rank) for rank in ranks]
        buffer_idxs = [self.heap[heap_idx].index for heap_idx in heap_idxs]
        encoded_sample = self.encode_samples(buffer_idxs, ranked_priority=True) # collect experience at given indices 
        return tuple(list(encoded_sample) + [weights, heap_idxs])
    
    # set the priorities of experiences at given indices 
    def update_priorities(self, heap_idxs, priorities):
        assert len(heap_idxs) == len(priorities)
        for idx, priority in zip(heap_idxs, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.heap)

            self.heap[idx].value = priority ** self.alpha 
            self.max_priority = max(self.max_priority, priority)

    # re-heapify. to be called periodically
    def sort(self):
        self.heap.build_heap()
        for i in range(len(self.heap)):
            buffer_idx = self.heap[i].index
            self.buffer[buffer_idx][1] = i      # update buffer's indices into heap
