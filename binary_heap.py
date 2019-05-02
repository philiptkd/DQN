from functools import total_ordering

# max heap
class Heap():
    def __init__(self, size):
        self.arr = [None]   # dummy
        self.max_size = size

    def __len__(self):
        return len(self.arr)-1

    def __getitem__(self, idx):
        assert idx+1 < len(self.arr)
        return self.arr[idx+1]

    def __setitem__(self, idx, val):
        assert idx+1 < len(self.arr)
        self.arr[idx+1] = val 

    def __repr__(self):
        return str(self.arr)

    # doesn't sort. use build_heap() periodically
    def insert(self, item): 
        if len(self.arr)-1 < self.max_size:
            self.arr.append(item)

    # doesn't assume any part of the tree is sorted
    def build_heap(self):
        for i in range((len(self.arr)-1)//2,0,-1):  # all the non-leaf nodes if the root is at arr[1]
            self._heapify(i)

    # turns array into heap
    # assumes subtrees are already heaps
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

    # removes and returns max value
    # should only ever be called from within get_kth_largest()
    def extract(self):
        last_idx = len(self.arr) - 1
        self._swap(1,last_idx)    # max now at last_idx
        max_item = self.arr.pop(last_idx)
        self._heapify(1)
        return max_item

    # O(k log k) time
    # returns index of kth largest item in heap
    def get_kth_largest(self, k):
        assert len(self.arr)-1 >= k
        if k==1:
            return 0

        # create new heap. add root of original heap
        aux_heap = Heap(k)
        aux_heap.insert(HeapItem(self.arr[1].value, 1)) # only need value and index into original heap
        
        for _ in range(k-1):
            aux_root = aux_heap.extract()   # delete root
            
            # add children
            left_child_idx = aux_root.index*2
            right_child_idx = left_child_idx + 1
            if left_child_idx < len(self.arr):
                aux_heap.insert(HeapItem(self.arr[left_child_idx].value, left_child_idx))
            if right_child_idx < len(self.arr):
                aux_heap.insert(HeapItem(self.arr[right_child_idx].value, right_child_idx))

        # the root is the kth largest element
        return aux_heap.arr[1].index - 1    # exposed indices are decremented by 1

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

    def __repr__(self):
        return "(value={0}, index={1})".format(self.value, self.index)
