import operator

# from https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
class SegmentTree():
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._tree_values = [neutral_element for _ in range(2*capacity)]
        self._operation = operation
        self._neutral = neutral_element

    # applies self.operation to contiguous subsequence of the array
    # this is not strictly needed since we only ever call this function with start=0 and end=capacity-1.
        # we only need to get value[0]. but this is more general.
    def range_query(self, start=0, end=None):
        if end is None:
            end = self._capacity-1
        return self._range_query_recur(start, end, 0, self._capacity-1, 1)

    def _range_query_recur(self, query_low, query_high, node_low, node_high, pos):
        #print("node range: [",node_low,",",node_high,"], query range: [",query_low,",",query_high,"], pos=",pos)

        # the query range encompasses this node's range
        if query_low <= node_low and query_high >= node_high:
            try:
                return self._tree_values[pos]
            except IndexError:
                print("Tried to access index",pos,"in array with",2*self._capacity,"elements.")
                raise IndexError

        # no overlap of ranges
        if query_low > node_high or query_high < node_low:
            return self._neutral    # this node should not contribute to the final answer

        # partial overlap
        mid = (node_low + node_high)//2
        return self._operation(
                    self._range_query_recur(query_low, query_high, node_low, mid, 2*pos),     # query left child 
                    self._range_query_recur(query_low, query_high, mid+1, node_high, 2*pos+1)     # query right child
                )

    def __setitem__(self, idx, val):
        idx += self._capacity   # index of the leaf
        self._tree_values[idx] = val
        idx //= 2   # go to parent
        while idx >= 1: # for every ancestor
            self._tree_values[idx] = self._operation(     # repair relationship with children
                self._tree_values[2 * idx],
                self._tree_values[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._tree_values[self._capacity + idx]    # second half of indices are the leaves


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).range_query(start, end)

    # for sampling 
    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum"""
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._tree_values[2 * idx] > prefixsum:    # if left child has greater sum
                idx = 2 * idx   # go to left child
            else:   # if sum of left child is too small
                prefixsum -= self._tree_values[2 * idx]   # look for remainder under right child
                idx = 2 * idx + 1
        return idx - self._capacity     # expose only the leaf index. also index into original array


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""
        return super(MinSegmentTree, self).range_query(start, end)

