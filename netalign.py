import numpy as np
from scipy.sparse import csr_matrix

from algorithms.graph import fullfill_with_ground_truth, load_csr_graphs
from algorithms.netalign import (netalign_main, netalign_setup,
                                 netalign_test_setup)
from data import get_sim_for_test, load_gts
from evaluation import evaluate

if __name__ == "__main__":
    (
        A,
        B,
        node_id_to_name_a,
        node_name_to_id_a,
        node_id_to_name_b,
        node_name_to_id_b,
    ) = load_csr_graphs("data/networks.txt")
    gt_train, gt_val, gt_test = load_gts(node_name_to_id_a, node_name_to_id_b)

    m, n = A.shape[0], B.shape[0]
    L = csr_matrix((m, n))
    L = fullfill_with_ground_truth(L, gt_train)
    L = fullfill_with_ground_truth(L, gt_val)
    L = netalign_test_setup(L, gt_test)
    S, LeWeights, li, lj = netalign_setup(A, B, L)
    
    data = {
        "S": S,
        "li": li,
        "lj": lj,
        "w": LeWeights,
    }
    sim = netalign_main(data)
    print(f'sim: {sim}')
    sim = get_sim_for_test(sim, gt_test)

    sim = sim.toarray() if not isinstance(sim, np.ndarray) else sim
    print(evaluate(sim, len(gt_test)))
