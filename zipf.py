import numpy as np
import pickle

# k ranges of equal probability for the zipf distribution on support 1,...,N
def quantiles(N,k,a):
    pmf = np.array([i**(-a) for i in range(1,N+1)])
    cdf = np.cumsum(pmf)
    cdf /= cdf[-1]
    quants = cdf//(1/k) # which of the k equally probable ranges does each point fall into

    ranges = [] # list of tuples (left,right) that are inclusive ranges of indices to sample from
    prev_range = (1,1)   # for use when no points fall into a range
    for i in range(k):
        where = np.nonzero(quants == i)[0]
        if len(where) > 0:
            this_range = (where[0]+1,where[-1]+1)   # ranges of ranks are 1-indexed
            ranges.append(this_range)
            prev_range = this_range
        else:
            ranges.append(prev_range)
    return ranges


def save_quantiles(N_list=None, k=20, alpha=0.6):
    if N_list is None:
        N_list = [k, 100,1000,5000,10000]

    quant_list = []
    for N in N_list:
        quant_list.append(quantiles(N,k,alpha))

    with open("quantiles.npy","wb") as f:
        pickle.dump((N_list, quant_list), f)


def load_quantiles():
    with open("quantiles.npy","rb") as f:
        N_list, quant_list = pickle.load(f)
    return N_list, quant_list

