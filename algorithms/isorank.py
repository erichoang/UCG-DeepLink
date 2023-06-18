import itertools
import math
from math import floor, log2

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse as sps

from . import bipartiteMatching
from .bipartiteMatching import (bipartite_matching_primal_dual,
                                          bipartite_matching_setup)
from .graph import fullfill_with_ground_truth

# original code https://github.com/constantinosskitsas/Framework_GraphAlignment


def edge_indicator(match, ei, ej):
    ind = np.zeros(len(ei), int)
    for i in range(0, len(ei)):
        if match[ei[i]] == ej[i]:
            ind[i] = 1
    return ind


def isorank(S, w, li, lj, a=0.5, b=1, alpha=2/3, rtype=2, tol=1e-12, maxiter=100, verbose=True):
    nzi = li.copy()
    nzi += 1
    nzi = np.insert(nzi, [0], [0])

    nzj = lj.copy()
    nzj += 1
    nzj = np.insert(nzj, [0], [0])

    ww = np.insert(w, [0], [0])

    m = max(li) + 1
    n = max(lj) + 1

    alpha = alpha if alpha else b/(a+b)

    # P is the normalized matrix S
    P = normout_rowstochastic(S)
    csum = math.fsum(w)
    v = w/csum
    nedges = np.shape(P)[0]
    allstats = True
    if allstats:
        rhistsize = 6
        row_pointers, column_indices, edge_values, tripi, _, _ = bipartite_matching_setup(
            None, nzi, nzj, ww, m, n)

        mperm1 = [x-1 for x in tripi if x > 0]
        mperm2 = [i for i, x in enumerate(tripi) if x > 0]
    else:
        rhistsize = 1
    r = alpha
    x = np.zeros(nedges, float) + v
    delta = 2
    it = 0
    reshist = np.zeros((maxiter+1, rhistsize), float)
    xbest = x
    fbest = 0
    fbestiter = 0
    if verbose and allstats:  # print the header
        print("{:5s}   {:4s}   {:8s}   {:7s} {:7s} {:7s} {:7}".format("best", "it",
                                                                      "pr-delta", "obj", "weight", "card", "overlap"))
    elif verbose:
        print("{:4s}   {:8s}", "iter", "delta")
    while it < maxiter and delta > tol:
        y = r * (P.T * x)
        omega = math.fsum(x) - math.fsum(y)
        y = y + omega * v
        delta = np.linalg.norm(x-y, 1)  # findd the correct one
        reshist[it] = delta
        it = it + 1
        x = y * (1/math.fsum(y))
        if allstats:
            if rtype == 1:
                xf = x
            elif rtype == 2:
                xf = a*v + b/2*(S*x)  # add v to preserve scale
            # ai = np.zeros(len(tripi), float)  # check the dimensions
            # ai[tripi > 0] = xf[mperm]
            edge_values = np.zeros(len(tripi))
            edge_values[mperm2] = xf[mperm1]
            edge_values = np.roll(edge_values, 1)
            _, _, _, noute1, match1 = bipartite_matching_primal_dual(
                row_pointers, column_indices, edge_values, tripi, m+1, n+1)
            ret_nodes_in_A = noute1-1
            # mi = bipartiteMatching.matching_indicator(
            #     rp, ci, match1, tripi, m, n)
            match1 = match1-1
            mi_int = edge_indicator(match1, li, lj)  # implement this
            # mi_int = mi[1:]
            val = np.dot(w, mi_int)
            overlap = np.dot(mi_int, (S*mi_int)/2)
            f = a*val + b*overlap
            # print(mi_int)
            # print(mi)
            if f > fbest:
                xbest = x
                fbest = f
                fbestiter = it
                itermark = "*"
            else:
                itermark = " "
            if verbose and allstats:
                print("{:5s}   {:4d}   {:8.1e}   {:5.2f} {:7.2f} {:7d} {:7d}".format(
                      itermark, it, delta, f, val, int(ret_nodes_in_A), int(overlap)))
                reshist[it, 1:-1] = [a*val + b*overlap, val, ret_nodes_in_A, overlap]
            elif verbose:
                print("{:4d}    {:8.1e}".format(it, delta))
    flag = delta > tol
    reshist = reshist[0:it, :]
    if allstats:
        x = xbest

    return x, nzi, nzj, m, n

def isorank_greedy(S, w, li, lj, a=0.5, b=1, alpha=2/3, rtype=2, tol=1e-12, maxiter=100, verbose=True):
    x, nzi, nzj, m, n= isorank(S, w, li, lj, a, b, alpha, rtype, tol, maxiter, verbose)

    sim = scipy.sparse.csr_matrix((x, (li, lj)), shape=(m, n))
    print(sim.toarray())

    xx = np.insert(x, [0], [0])
    m, n, val, noute, match1 = bipartiteMatching.bipartite_matching(
        None, nzi, nzj, xx)
    ret_nodes_in_A, ret_matched_nodes_in_B = bipartiteMatching.edge_list(m, n, val, noute, match1)

    return ret_nodes_in_A, ret_matched_nodes_in_B


def normout_rowstochastic(S):  # to check
    """
    This function normalizes the rows of a matrix to sum to 1
    """

    n = np.shape(S)[0]
    m = np.shape(S)[1]
    colsums = S.sum(1)

    pi, pj, pv = scipy.sparse.find(S)

    D = colsums[pi].T
    x1 = np.true_divide(pv, D)
    x = np.ravel(x1)
    Q = scipy.sparse.csr_matrix((x, (pi, pj)), shape=(m, n))

    return Q


def create_S(A:sps.csr_matrix, B:sps.csr_matrix, L:sps.csr_matrix) -> sps.csr_matrix:
    """
    CSR: Compressed Sparse Row,
        where indptr: row indices, indices: column indices, data: values
    In the paper of Global alignment of multiple protein interaction networks with application to functional orthology detection (2013)
    S is defined as:
    S = [Sij] = [1 if (i, j) in E(A) and (i', j') in E(B) and L(i, i') > 0 else 0]
    """
    # return None
    n = A.shape[0]
    m = B.shape[0]

    # the column indices for row i are stored in indices[indptr[i]:indptr[i+1]]
    csr_pointers, nodes_in_B = L.indptr, L.indices
    # nodes_in_A == nodes_in_B since each element from one group together indicates a correspondance between nodes from A and B
    nedges = len(nodes_in_B)

    Si = []
    Sj = []
    # a vector in the length of m (number of nodes in B), initialized by -1
    wv = np.full(m, -1)
    ri1 = 0
    for i in range(n):
        for ri1 in range(csr_pointers[i], csr_pointers[i+1]):
            # rages through pointers of all neighbors of node i in L
            # nodes_in_B[ri1] indicates the corresponding node in B
            wv[nodes_in_B[ri1]] = ri1

        for ip in A[i].nonzero()[1]:
            # in graph A, for each neighbor of node i (except i itself)
            if i == ip:
                continue
            for ri2 in range(csr_pointers[ip], csr_pointers[ip+1]):
                jp = nodes_in_B[ri2]
                for j in B[jp].nonzero()[1]:
                    if j == jp:
                        continue
                    if wv[j] >= 0:
                        Si.append(ri2)
                        Sj.append(wv[j])
        for ri1 in range(csr_pointers[i], csr_pointers[i+1]):
            wv[nodes_in_B[ri1]] = -1

    return sps.csr_matrix(([1]*len(Si), (Sj, Si)), shape=(nedges, nedges), dtype=int)


def create_L(A: np.ndarray, B, lalpha=1, min_similarity=None) -> sps.csr_matrix:
    """
    In the paper of IsoRank, the matrix L indicates the bipartite graph between the nodes in the graph A and the nodes in B.
    According to the paper, L is a matrix of size n x m, where n is the number of nodes in A and m is the number of nodes in B.
    Each row of L is a probability distribution over the nodes in B.
    The probability distribution is defined as follows:
    For each node i in A, we sort the nodes in B according to their degrees.
    We then take such nodes from B, whose degrees are mostly similar to their correspondance in A and denote they is an one-on-many
     match. The band with of tolerance is defined using floor(lalpha * log2(m)).
    The rest of the nodes in B are assigned a probability of 0.
    :param A: The adjacency matrix of the first graph.
    :param B: The adjacency matrix of the second graph.
    :param lalpha: The parameter lalpha.
    :param mind: The minimum degree of a node in A.
    :return: The matrix L.
    """
    n = A.shape[0]
    m = B.shape[0]

    if lalpha is None:
        return sps.csr_matrix(np.ones((n, m)))

    a = A.sum(1)
    b = B.sum(1)

    node_idx_and_degrees_a = list(enumerate(a))
    node_idx_and_degrees_a.sort(key=lambda x: x[1])

    node_idx_and_degrees_b = list(enumerate(b))
    node_idx_and_degrees_b.sort(key=lambda x: x[1])

    ab_correspondance = [0] * n
    start = 0
    end = floor(lalpha * log2(m))
    for node_idx_a, node_degree_a in node_idx_and_degrees_a:
        # node_idx_and_degrees_a[:, 1] = 1 2 3 4 5 6 7
        # ap[1]                                ^
        #                                s   e  
        # in this case,  abs(node_idx_and_degrees_b[end][1] - ap[1]) <= abs(node_idx_and_degrees_b[start][1] - ap[1])
        #
        # node_idx_and_degrees_a[:, 1] = 1 2 3 4 5 6 7
        # ap[1]                                ^
        #                                      s   e
        # the while loop will stop at this point and the ab_m[ap[0]] will be assigned to b's nodes corresponding to [5, 6, 7]
        while(end < m and
              abs(node_idx_and_degrees_b[end][1] - node_degree_a) <= abs(node_idx_and_degrees_b[start][1] - node_degree_a)
              ):
            end += 1
            start += 1
        ab_correspondance[node_idx_a] = [bp[0] for bp in node_idx_and_degrees_b[start:end]]

    graph_1_nodes = []
    graph_2_nodes = []
    similarity = []
    for i, bj in enumerate(ab_correspondance):
        for j in bj:
            sim_score = 0.001
            # d = 1 - abs(a[i]-b[j]) / max(a[i], b[j])
            if min_similarity is None:
                if sim_score > 0:
                    graph_1_nodes.append(i)
                    graph_2_nodes.append(j)
                    similarity.append(sim_score)
            else:
                graph_1_nodes.append(i)
                graph_2_nodes.append(j)
                similarity.append(min_similarity if sim_score <= 0 else sim_score)

    return sps.csr_matrix((similarity, (graph_1_nodes, graph_2_nodes)), shape=(n, m))
