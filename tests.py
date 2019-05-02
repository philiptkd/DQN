from experience_replay import RankBasedReplay
from zipf import save_quantiles
from binary_heap import Heap, HeapItem

alpha = 1
beta = 1
max_size = 10
k = 5

def test():
    save_quantiles(k=k, alpha=alpha)
    replay = RankBasedReplay(max_size, alpha)
    for i in range(5):
        replay.add((i,i,i,i,i))

    replay.update_priorities([0,1,2,3,4], [0.5,1,2,3,4])
    replay.sort()

    print(replay.buffer)
    print(replay.heap)

    s,a,r,s,d,w,i = replay.sample(k, beta)
    print([(s,i) for s,i in zip(s,i)])

def test2():
    heap = Heap(max_size)
    for i in range(5):
        heap.insert(HeapItem(i,i))
    print(heap)
    heap.build_heap()
    print(heap)


test()
