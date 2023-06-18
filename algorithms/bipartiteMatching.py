import scipy
import numpy as np


def bipartite_matching(A, nzi, nzj, nzv):
    # return bipartite_matching_primal_dual(bipartite_matching_setup(A,nzi,nzj,nzv))
    rp, ci, ai, tripi, m, n = bipartite_matching_setup(A, nzi, nzj, nzv)
    # print("hi7")
    return bipartite_matching_primal_dual(rp, ci, ai, tripi, m, n)


def bipartite_matching_primal_dual(row_pointers, column_indices, edge_weights, tripi, m, n):
    """
    This function solves the bipartite matching problem using the primal-dual algorithm.
    
    (Seems like a hungarian algorithm?)

    Returns the matching indicator and the matching weight.
    Returns:
    - match1: the matching indicator, match[i] = j means i is matched to j
    - noute: the count of matched nodes
    """
    # print(m, n)
    # print(rp.tolist())
    # print(ci.tolist())
    # print(ai.tolist())
    # print(tripi.tolist())
    # variables used for the primal-dual algorithm
    # normalize ai values # updated on 2-19-2019
    edge_weights = edge_weights/np.amax(abs(edge_weights))
    alpha = np.zeros(m)
    bt = np.zeros(m+n)  # beta
    queue = np.zeros(m, int)
    t = np.zeros(m+n, int)
    match1 = np.zeros(m, int)
    match2 = np.zeros(m+n, int)
    tmod = np.zeros(m+n, int)
    ntmod = 0

    # initialize the primal and dual variables

    for i in range(1, m):
        for rpi in range(row_pointers[i], row_pointers[i+1]):
            if edge_weights[rpi] > alpha[i]:
                alpha[i] = edge_weights[rpi]

    # dual variables (bt) are initialized to 0 already
    # match1 and match2 are both 0, which indicates no matches

    i = 1
    while i < m:
        for j in range(1, ntmod+1):
            t[tmod[j]] = 0
        ntmod = 0
        # add i to the stack
        head = 1
        tail = 1
        queue[head] = i
        while head <= tail and match1[i] == 0:
            k = queue[head]
            for rpi in range(row_pointers[k], row_pointers[k+1]):
                j = column_indices[rpi]
                if edge_weights[rpi] < alpha[k] + bt[j] - 1e-8:
                    continue

                if t[j] == 0:
                    tail = tail+1
                    if tail < m:
                        queue[tail] = match2[j]
                    t[j] = k
                    ntmod = ntmod+1
                    tmod[ntmod] = j
                    if match2[j] < 1:
                        while j > 0:
                            match2[j] = t[j]
                            k = t[j]
                            temp = match1[k]
                            match1[k] = j
                            j = temp
                        break
            head = head+1
        if match1[i] < 1:
            theta = np.inf
            for j in range(1, head):
                t1 = queue[j]
                for rpi in range(row_pointers[t1], row_pointers[t1+1]):
                    t2 = column_indices[rpi]
                    if t[t2] == 0 and alpha[t1] + bt[t2] - edge_weights[rpi] < theta:
                        theta = alpha[t1] + bt[t2] - edge_weights[rpi]
            for j in range(1, head):
                alpha[queue[j]] -= theta
            for j in range(1, ntmod+1):
                bt[tmod[j]] += theta
            continue
        i = i+1
        # print(f"{i} < {m}")
    val = 0
    # print("po")
    for i in range(1, m):
        for rpi in range(row_pointers[i], row_pointers[i+1]):
            if column_indices[rpi] == match1[i]:
                val = val+edge_weights[rpi]
    noute = 0
    for i in range(1, m):
        if match1[i] <= n:
            noute = noute+1
    return m, n, val, noute, match1


def bipartite_matching_setup(A, non_zero_row_idxs, nzj, nzv, m=None, n=None):
    """
    This function sets up the bipartite matching problem. 
    It takes in a sparse matrix A and returns the data structures needed for the primal-dual algorithm.

    nzi, nzj, nzv are the row indices, column indices, and values of the nonzero entries of A, respectively.

    Args:
        A (scipy.sparse.csr_matrix): sparse matrix
        nzi (list): list of row indices
        nzj (list): list of column indices
        nzv (list): list of values, suggest the current matching affinity of the correponding pair of nodes
        m (int): number of rows
        n (int): number of columns
        
    Returns:
        rp (list): list of row pointers
        ci (list): list of column indices
        ai (list): list of values
        
        the three lists above are used to represent the sparse matrix A in the CSR format.

        tripi (list): index of the edges in the primal graph, -1 for the extra edges added in the setup phase
        m (int): number of rows
        n (int): number of columns
    """
    # (nzi,nzj,nzv) = bipartite_matching_setup_phase1(A,nzi,nzj,nzv)
    if A is not None:
        (non_zero_row_idxs, nzj, nzv) = get_non_zero_indices_and_values(A)
        (m, n) = np.shape(A)
        m = m+1  # ?
        n = n+1  # ?
    if m is None:
        m = max(non_zero_row_idxs) + 1
    if n is None:
        n = max(nzj) + 1
    # print(nzi)
    # print(nzj)
    # print(nzv)
    # print(m, n)
    # print("hi-setup")
    nedges = len(non_zero_row_idxs)
    rp = np.ones(m+2, int)  # csr matrix with extra edges
    # the reason why here it is nedges + m is because we need to add extra edges to make the graph balanced
    ci = np.zeros(nedges+m, int)
    ai = np.zeros(nedges+m)
    tripi = np.zeros(nedges+m, int)

    rp[0] = 0
    rp[1] = 0
    for i in range(1, nedges):
        rp[non_zero_row_idxs[i]+1] = rp[non_zero_row_idxs[i]+1]+1  # pointers (in CSR sparse matrix format) for each row
    rp = np.cumsum(rp)

    for i in range(1, nedges):
        tripi[rp[non_zero_row_idxs[i]]+1] = i
        ci[rp[non_zero_row_idxs[i]]+1] = nzj[i]
        ai[rp[non_zero_row_idxs[i]]+1] = nzv[i]
        rp[non_zero_row_idxs[i]] = rp[non_zero_row_idxs[i]]+1

    for i in range(1, m+1):  # add the extra edges
        tripi[rp[i]+1] = -1
        ai[rp[i]+1] = 0
        ci[rp[i]+1] = n+i
        rp[i] = rp[i]+1

    # restore the row pointer array
    for i in range(m, 0, -1):
        rp[i+1] = rp[i]
    rp[1] = 0
    rp = rp+1

    # check for duplicates in the data
    colind = np.zeros(m+n, int)
    for i in range(1, m):
        for rpi in range(rp[i], rp[i+1]):
            if colind[ci[rpi]] == 1:
                print("bipartite_matching:duplicateEdge")
        colind[ci[rpi]] = 1

        for rpi in range(rp[i], rp[i+1]):
            colind[ci[rpi]] = 0
    return rp, ci, ai, tripi, m, n


def get_non_zero_indices_and_values(A, nzi, nzj, nzv):
    temp = len(nzi)+1
    nzi1 = np.zeros(temp, int)
    nzj1 = np.zeros(temp, int)
    nzv1 = np.zeros(temp, float)
    for i in range(1, temp):
        nzi1[i] = nzi[i-1]
        nzj1[i] = nzj[i - 1]
        nzv1[i] = nzv[i - 1]
    nzi1 = nzi1+1
    nzj1 = nzj1+1
    return (nzi1, nzj1, nzv1)


def get_non_zero_indices_and_values(A):
    '''
    This function takes in a sparse matrix A and returns the row indices, column indices, 
    and values of the nonzero entries of A, respectively.

    Note that the row and column indices are 1-indexed.
    
    Args:
        A (scipy.sparse.csr_matrix): sparse matrix
        
    Returns:
        nzi1 (list): list of row indices
        nzj1 (list): list of column indices
        nzv1 (list): list of values
    '''
    nzi, nzj = scipy.sparse.csr_matrix.nonzero(A)
    temp = len(nzi) + 1
    nzi1 = np.zeros(temp, int)
    nzj1 = np.zeros(temp, int)
    nzv1 = np.zeros(temp, float)
    for i in range(1, temp):
        nzi1[i] = nzi[i - 1]
        nzj1[i] = nzj[i - 1]
        nzv1[i] = A[nzi1[i], nzj1[i]]
    nzi1 = nzi1 + 1
    nzj1 = nzj1 + 1

    return (nzi1, nzj1, nzv1)


def edge_list(m, n, weight, cardinality, match):
    m1 = np.zeros(cardinality, int)
    m2 = np.zeros(cardinality, int)
    noute = 0
    for i in range(1, m):
        if match[i] <= n:
            m1[noute] = i
            m2[noute] = match[i]
            noute = noute+1
    return m1, m2


def matching_indicator(rp, ci, match1, tripi, m, n):
    mi = np.zeros(len(tripi)-m, int)
    for i in range(1, m+1):
        for rpi in range(rp[i], rp[i+1]):
            if match1[i] <= n and ci[rpi] == match1[i]:
                mi[tripi[rpi]] = 1
    return mi
