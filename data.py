import numpy as np


def get_sim_for_test(old_sim, gt_test) -> np.ndarray:
    temp = sorted([[a, b] for a, b in gt_test.items()], key=lambda x: x[1])
    a_indices, b_indices = list(zip(*temp))
    return (old_sim[a_indices, :])[:, b_indices].toarray()


def load_gts(node_name_to_id_a, node_name_to_id_b):
    with open("data/gts.txt", "r") as f:
        lines = f.readlines()
    gt_train = [
        [node_name_to_id_a[a], node_name_to_id_b[b]] for a, b in eval(lines[0].strip())
    ]
    gt_val = [
        [node_name_to_id_a[a], node_name_to_id_b[b]] for a, b in eval(lines[1].strip())
    ]

    # gt_val = {node_name_to_id_a[a]: node_name_to_id_b[b] for a, b in eval(lines[1].strip())}
    gt_test = {
        node_name_to_id_a[a]: node_name_to_id_b[b] for a, b in eval(lines[2].strip())
    }
    return gt_train, gt_val, gt_test


def numerate(node_names, nodes_name_to_id):
    return [nodes_name_to_id[name] for name in node_names]