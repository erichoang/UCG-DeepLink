from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import scipy
from scipy.sparse import csr_matrix

from data import numerate


def convert_edge_list_to_csr(Ai, Aj) -> scipy.sparse.csr_matrix:
    n = max(max(Ai), max(Aj)) + 1
    nedges = len(Ai)
    Aw = np.ones(nedges)
    A = csr_matrix((Aw, (Ai, Aj)), shape=(n, n), dtype=int)
    A = A + A.T  # make undirected
    return A


def load_csr_from_json_string(
    json_string,
) -> Tuple[csr_matrix, List[str], Dict[str, int]]:
    temp = eval(json_string)
    Ai, Aj = list(zip(*[[_["source"], _["target"]] for _ in temp["edges"]]))
    nodes_id_to_name = sorted(list(set(Ai).union(set(Aj))))
    nodes_name_to_id = defaultdict(int)
    for i, name in enumerate(nodes_id_to_name):
        nodes_name_to_id[name] = i
    Ai = numerate(Ai, nodes_name_to_id)
    Aj = numerate(Aj, nodes_name_to_id)
    return convert_edge_list_to_csr(Ai, Aj), nodes_id_to_name, nodes_name_to_id


def load_csr_graphs(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
    A, node_id_to_name_a, node_name_to_id_a = load_csr_from_json_string(
        lines[0].strip()
    )
    B, node_id_to_name_b, node_name_to_id_b = load_csr_from_json_string(
        lines[1].strip()
    )

    return (
        A,
        B,
        node_id_to_name_a,
        node_name_to_id_a,
        node_id_to_name_b,
        node_name_to_id_b,
    )


def fullfill_with_ground_truth(L, gt, value=1):
    for a, b in gt:
        L[a, b] = value
    return L