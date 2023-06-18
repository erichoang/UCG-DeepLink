import scipy

from algorithms import isorank
from algorithms.graph import fullfill_with_ground_truth, load_csr_graphs
from data import get_sim_for_test, load_gts
from evaluation import evaluate

a, b = 0.2, 0.8


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
    
    L = isorank.create_L(A, B, lalpha=2)
    L = fullfill_with_ground_truth(L, gt_train)
    L = fullfill_with_ground_truth(L, gt_val)
    S = isorank.create_S(A, B, L)

    graph_1_nodes, graph_2_nodes, w = scipy.sparse.find(L)
    x, nzi, nzj, m, n = isorank.isorank(S, w, graph_1_nodes, graph_2_nodes, a, b)
    sim = scipy.sparse.csr_matrix((x, (graph_1_nodes, graph_2_nodes)), shape=(m, n))
    sim = get_sim_for_test(sim, gt_test)

    print(evaluate(sim, len(gt_test)))
    