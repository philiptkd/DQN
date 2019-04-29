import math
from experience_replay import PrioritizedReplay


# assumes tree size is power of 2
def print_tree(tree_array):
    tree_array = [int(x) for x in tree_array]
    length = len(tree_array)
    assert length & (length - 1) == 0   # is power of 2
    
    height = int(math.log(length, 2))

    num_spaces = 0
    for level in range(height,0,-1):
        print("    "*num_spaces, end="")
        if level == 1:
            print("  ", end="")

        width = 1 << level-1
        for i in range(width):
            print(str(tree_array[width+i])+"   ", end="")
        print('\n')
        
        num_spaces += width >> 2


def test_replay():
    replay = PrioritizedReplay(5,1)
    for i in range(5):
        replay.add((i,i,i,i,0))

    print_tree(replay._sum_tree._tree_values)
    s,a,r,s,d,w,i = replay.sample(2,1)
    print(s,a,r,s,d,w,i)

    replay.update_priorities(i,[100,3])
    print_tree(replay._sum_tree._tree_values)

    for _ in range(10):
        s,a,r,s,d,w,i = replay.sample(2,1)
        print(s,a,r,s,d,w,i)
        

test_replay()
