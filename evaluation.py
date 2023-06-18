import numpy as np
from sklearn.metrics import roc_auc_score


def mrr(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred)
    return np.mean(1 / (np.argwhere(y_true == y_pred)[:, 1] + 1))


def success_at_k(y_true, y_pred, k=2):
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred)
    return np.mean(np.any(y_true == y_pred[:, :k], axis=1))


def evaluate(Y_pred_cls, len_targets):
    def convert_to_ranking(pred_cls):
        return pred_cls.argsort(axis=1)[:, ::-1]

    Y_label_cls = np.eye(len_targets)

    Y_label_ranking = list(range(len_targets))
    Y_pred_ranking = convert_to_ranking(Y_pred_cls)

    result = {
        "AUC": roc_auc_score(Y_label_cls, Y_pred_cls),
        # MRR is equivalent to MAP in this case, because the ranking is binary
        "MRR": mrr(Y_label_ranking, Y_pred_ranking),
        "Success_at_k": success_at_k(Y_label_ranking, Y_pred_ranking, k=2),
    }
    return result
