import itertools
import numpy as np
import scipy.sparse as sps
from scipy.sparse import csc_matrix, csr_matrix

from .graph import fullfill_with_ground_truth

from . import bipartitewrapper as bmw

# original code https://www.cs.purdue.edu/homes/dgleich/codes/netalign/, https://github.com/constantinosskitsas/Framework_GraphAlignment


def make_squares(A: csr_matrix, B: csr_matrix, L: csr_matrix):
    m = np.shape(A)[0]
    n = np.shape(B)[0]

    rpA = A.indptr
    ciA = A.indices
    rpB = B.indptr
    ciB = B.indices

    rpAB = L.indptr
    ciAB = L.indices
    vAB = L.data

    Se1 = []
    Se2 = []

    wv = -np.ones(n, int)
    for i in range(m):
        possible_matches = list(range(rpAB[i], rpAB[i + 1]))
        wv[ciAB[possible_matches]] = possible_matches
        for ri1 in range(rpA[i], rpA[i + 1]):
            ip = ciA[ri1]
            if i == ip:
                continue
            for ri2 in range(rpAB[ip], rpAB[ip + 1]):
                if ri2 == 9:
                    a = 1
                jp = ciAB[ri2]
                for ri3 in range(rpB[jp], rpB[jp + 1]):
                    j = ciB[ri3]
                    if j == jp:
                        continue
                    if wv[j] >= 0:
                        Se1.append(ri2)
                        Se2.append(wv[j])
        wv[ciAB[possible_matches]] = -1

    Le = np.zeros((L.getnnz(), 2), int)
    LeWeights = np.zeros(L.getnnz(), float)
    for i in range(0, m):
        j = list(range(rpAB[i], rpAB[i + 1]))
        Le[j, 0] = i
        Le[j, 1] = ciAB[j]
        LeWeights[j] = vAB[j]
    Se = np.zeros((len(Se1), 2), int)
    Se[:, 0] = Se1
    Se[:, 1] = Se2
    return Se, Le, LeWeights


def netalign_setup(A: csr_matrix, B: csr_matrix, L: csr_matrix):  # needs fix
    Se, Le, LeWeights = make_squares(A, B, L)
    li = Le[:, 0]
    lj = Le[:, 1]
   
    Se1 = Se[:, 0]
    Se2 = Se[:, 1]
    values = np.ones(len(Se1))
    el = L.getnnz()
    S = csc_matrix(
        (
            values,
            (
                Se1,
                Se2,
            ),
        ),
        (el, el),
    )

    return S, LeWeights, li, lj


def othermaxplus(dim, li, lj, lw, m, n):
    if dim == 1:
        i1 = lj
        i2 = li
        N = n
    else:
        i1 = li
        i2 = lj
        N = m

    dimmax1 = np.zeros(N)
    dimmax2 = np.zeros(N)
    dimmaxind = np.zeros(N)
    nedges = len(li)

    for i in range(nedges):
        if lw[i] > dimmax2[i1[i]]:
            if lw[i] > dimmax1[i1[i]]:
                dimmax2[i1[i]] = dimmax1[i1[i]]
                dimmax1[i1[i]] = lw[i]
                dimmaxind[i1[i]] = i2[i]
            else:
                dimmax2[i1[i]] = lw[i]

    omp = np.zeros(len(lw))
    for i in range(nedges):
        if i2[i] == dimmaxind[i1[i]]:
            omp[i] = dimmax2[i1[i]]
        else:
            omp[i] = dimmax1[i1[i]]

    return omp


def othersum(si, sj, s, m, n):
    rowsum = accumarray(si, s, m)
    return rowsum[si] - s


def accumarray(xij, xw, n):
    sums = np.zeros(n)

    for i in range(len(xij)):
        sums[xij[i]] += xw[i]

    return sums


def netalign_main(data, a=1, b=1, gamma=0.99, dtype=2, maxiter=100, verbose=True):
    """
    NetAlign algorithm
    :param data: dictionary with keys 'S', 'li', 'lj', 'w'
        key 'S': square matrix S
        key 'li': list of indices of rows of S
        key 'lj': list of indices of columns of S
        key 'w': list of weights of edges
    :param a: alpha
    :param b: beta
    :param gamma: gamma
    :param dtype: 1 for maxplus, 2 for maxsum
    :param maxiter: maximum number of iterations
    :param verbose: print progress
    :return: dictionary with keys 'x', 'y', 'objective', 'iter', 'time'
    """

    S = data["S"]
    li = data["li"]
    lj = data["lj"]
    w = data["w"]

    S = sps.csr_matrix(S)

    nedges = len(li)
    nsquares = S.count_nonzero() // 2
    
    # compute a vector that allows us to transpose data between squares.
    sui, suj, _ = sps.find(sps.triu(S, 1))
    SI = sps.csr_matrix((list(range(1, len(sui) + 1)), (sui, suj)), shape=S.shape)

    SI = SI + SI.transpose()
    si, sj, sind = sps.find(SI)
    sind = [x - 1 for x in sind]
    SP = sps.csr_matrix(([1] * len(si), (si, sind)), shape=(S.shape[0], nsquares))
    sij, sijrs, _ = sps.find(SP)
    sind = list(range(SP.count_nonzero()))
    spair = sind[:]
    spair[::2] = sind[1::2]
    spair[1::2] = sind[::2]

    # Initialize the messages
    ma = np.zeros(nedges, int)
    mb = np.zeros(nedges, int)
    ms = np.zeros(S.count_nonzero())
    sums = np.zeros(nedges)

    damping = gamma
    curdamp = 1
    alpha = a
    beta = b

    fbest = 0
    fbestiter = 0
    if verbose:
        print(
            "{:4s}   {:>4s}   {:>7s} {:>7s} {:>7s} {:>7s}   {:>7s} {:>7s} {:>7s} {:>7s}".format(
                "best",
                "iter",
                "obj_ma",
                "wght_ma",
                "card_ma",
                "over_ma",
                "obj_mb",
                "wght_mb",
                "card_mb",
                "over_mb",
            )
        )

    setup, m, n = bmw.bipartite_setup(li, lj, w)

    for it in range(1, maxiter + 1):
        prevma = ma
        prevmb = mb
        prevms = ms
        prevsums = sums
        curdamp = damping * curdamp

        omaxb = np.array(
            [max(0, x) for x in othermaxplus(2, li, lj, mb, m, n)],
        )
        omaxa = np.array(
            [max(0, x) for x in othermaxplus(1, li, lj, ma, m, n)],
        )

        msflip = ms[spair]
        mymsflip = msflip + beta
        mymsflip = [min(beta, x) for x in mymsflip]
        mymsflip = [max(0, x) for x in mymsflip]
        # mymsflip is F in the paper

        sums = accumarray(sij, mymsflip, nedges)
        # d^(t) in the paper

        ma = alpha * w - omaxb + sums
        mb = alpha * w - omaxa + sums

        ms = alpha * w[sij] - (omaxb[sij] + omaxa[sij])
        ms += othersum(sij, sijrs, mymsflip, nedges, nsquares)

        if dtype == 1:
            ma = curdamp * (ma) + (1 - curdamp) * (prevma)
            mb = curdamp * (mb) + (1 - curdamp) * (prevmb)
            ms = curdamp * (ms) + (1 - curdamp) * (prevms)
        elif dtype == 2:
            ma = ma + (1 - curdamp) * (prevma + prevmb - alpha * w + prevsums)
            mb = mb + (1 - curdamp) * (prevmb + prevma - alpha * w + prevsums)
            ms = ms + (1 - curdamp) * (prevms + prevms[spair] - beta)
        elif dtype == 3:
            ma = curdamp * ma + (1 - curdamp) * (prevma + prevmb - alpha * w + prevsums)
            mb = curdamp * mb + (1 - curdamp) * (prevmb + prevma - alpha * w + prevsums)
            ms = curdamp * ms + (1 - curdamp) * (prevms + prevms[spair] - beta)

        hista = bmw.round_messages(ma, S, w, alpha, beta, setup, m, n)[:-2]

        histb = bmw.round_messages(mb, S, w, alpha, beta, setup, m, n)[:-2]

        if hista[0] > fbest:
            fbestiter = it
            mbest = ma
            fbest = hista[0]
        if histb[0] > fbest:
            fbestiter = -it
            mbest = mb
            fbest = histb[0]

        if verbose:
            if fbestiter == it:
                bestchar = "*a"
            elif fbestiter == -it:
                bestchar = "*b"
            else:
                bestchar = ""
            print(
                "{:4s}   {:4d}   {:7.2f} {:7.2f} {:7d} {:7d}   {:7.2f} {:7.2f} {:7d} {:7d}".format(
                    bestchar, it, *hista, *histb
                )
            )

    return sps.csr_matrix((mbest, (li, lj)))


def netalign_test_setup(L, gt_test):
    temp = sorted([[a, b] for a, b in gt_test.items()], key=lambda x: x[1])
    prod_test = list(itertools.product(*zip(*temp)))
    L = fullfill_with_ground_truth(L, prod_test, 1./len(gt_test))
    return L

