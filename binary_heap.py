class Heap():
    def __init__(self, size):
        self.arr = []
        self.max_size = size
        self.next_idx = 1   # start indexing at 1

    def insert(self, item):
        if len(self.arr) < self.max_size:
            self.arr.append(item)
        else:
            self.arr[self.next_idx] = item
        self.next_idx = (self.next_idx + 1)%self.max_size

    # doesn't assume any part of the tree is sorted
    def build_heap(self):
        for i in range((len(self.arr)-1)//2,0,-1):  # all the non-leaf nodes if the root is at arr[1]
            self._heapify(i)

    def _heapify(self, idx):
        left_idx = 2*idx
        right_idx = 2*idx + 1

        if left_idx < len(self.arr) and self.arr[left_idx] > self.arr[idx]:
            greatest_idx = left_idx
        else:
            greatest_idx = idx
        if right_idx < len(self.arr) and self.arr[right_idx] > self.arr[greatest_idx]:
            greatest_idx = right_idx

        if greatest_idx != idx:
            self._swap(greatest_idx, idx)
            self._heapify(greatest_idx)

    def _swap(self, i, j):
        tmp = self.arr[i]
        self.arr[i] = self.arr[j]
        self.arr[j] = tmp

    # TODO: calculate segment boundaries, sample from each segment
    def sample(self, batch_size):
        pass

    def update_priorities(self, priorities, indxs):
        pass

# for inserting into heap.
# index is into the replay buffer and refers to a specific state transition
@total_ordering     # generates missing compare methods
class HeapItem():
    def __init__(self, value, index):
        self.value = value
        self.index = index

    def __eq__(self, other):
        return self.value == other.value

    def __gt__(self, other):
        return self.value > other.value
