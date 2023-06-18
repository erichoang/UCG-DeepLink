import json
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

from algorithms.graph import load_csr_graphs
from data import load_gts
from evaluation import evaluate


def load_prior_embeddings():
    ebd_fname = "data/individuals_info+word.json"
    with open(ebd_fname, 'r') as fin:
        loaded = json.load(fin)
    embeddings = {k: {kk: np.array(vv['metadata']['embedding']) for kk, vv in v.items()} for k, v in loaded.items()}
    return embeddings["roxdv3"], embeddings["roxhood"]


def tf_idf_analyzer(gt_train: List[List[str]], gt_test: Dict[str, str], 
                    node_id_to_name_a: List[str], node_id_to_name_b: List[str]):
    prior_ebd1, prior_ebd2 = load_prior_embeddings()

    train_labels = sorted(gt_train)
    test_labels = sorted(list(gt_test.items()))
    X_train = [prior_ebd1[node_id_to_name_a[node1]] for node1, _ in train_labels]
    Y_train = [prior_ebd2[node_id_to_name_b[node2]] for _, node2 in train_labels]
    
    model = LinearRegression().fit(X_train, Y_train)
    
    X_test = [prior_ebd1[node_id_to_name_a[node1]] for node1, _ in test_labels]
    Y_test = [prior_ebd2[node_id_to_name_b[node2]] for _, node2 in test_labels]
    Y_hat = model.predict(X_test)

    Y_pred_cls = cosine_similarity(Y_hat, Y_test)    
    return Y_pred_cls


if __name__ == "__main__":
    (
        _,
        _,
        node_id_to_name_a,
        node_name_to_id_a,
        node_id_to_name_b,
        node_name_to_id_b,
    ) = load_csr_graphs("data/networks.txt")
    gt_train, gt_val, gt_test = load_gts(node_name_to_id_a, node_name_to_id_b)

    sim = tf_idf_analyzer(gt_train, gt_test, node_id_to_name_a, node_id_to_name_b)

    sim = sim.toarray() if not isinstance(sim, np.ndarray) else sim
    print(evaluate(sim, len(gt_test)))
