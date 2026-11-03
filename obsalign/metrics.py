"""Classification and calibration metrics.

Undefined AUROC is represented as JSON null with an explicit reason.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score


def expected_calibration_error(confidence, correct, weights=None, bins=15):
    confidence, correct = np.asarray(confidence), np.asarray(correct)
    weights = np.ones_like(confidence, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("ECE needs positive total weight")
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    result = 0.0
    for b in range(bins):
        use = index == b
        mass = weights[use].sum()
        if mass:
            acc = np.average(correct[use], weights=weights[use])
            conf = np.average(confidence[use], weights=weights[use])
            result += mass / total * abs(acc - conf)
    return float(result)


def classification_metrics(y, probabilities, task, weights=None):
    y, p = np.asarray(y), np.asarray(probabilities)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    use = w > 0
    y, p, w = y[use], p[use], w[use]
    if len(y) == 0:
        return {"status": "undefined", "reason": "no positive-weight samples", "n": 0}
    c = p.shape[1]
    if task == "multiclass":
        pred = p.argmax(1)
        accuracy = accuracy_score(y, pred, sample_weight=w)
        f1 = f1_score(y, pred, labels=np.arange(c), average="macro", sample_weight=w, zero_division=0)
        recall = recall_score(y, pred, labels=np.arange(c), average="macro", sample_weight=w, zero_division=0)
        nll = np.average(-np.log(np.clip(p[np.arange(len(y)), y.astype(int)], 1e-12, 1)), weights=w)
        ece = expected_calibration_error(p.max(1), pred == y, w)
        if len(np.unique(y)) == c:
            auc = roc_auc_score(y, p[:, 1], sample_weight=w) if c == 2 else roc_auc_score(
                y, p, labels=np.arange(c), average="macro", multi_class="ovr", sample_weight=w)
            reason = None
        else:
            auc, reason = None, "AUROC requires all task classes in this evaluation view"
        per_label_auc = None
    else:
        pred = p >= 0.5
        accuracy = accuracy_score(y, pred, sample_weight=w)
        f1 = f1_score(y, pred, average="macro", sample_weight=w, zero_division=0)
        recall = recall_score(y, pred, average="macro", sample_weight=w, zero_division=0)
        nll = np.average(-(y * np.log(p.clip(1e-12, 1)) + (1-y) * np.log((1-p).clip(1e-12, 1))).mean(1), weights=w)
        # Binary decision ECE over all sample-label decisions.
        ece = expected_calibration_error(np.maximum(p, 1-p).ravel(), (pred == y).ravel(), np.repeat(w, c))
        per_label_auc = [float(roc_auc_score(y[:, k], p[:, k], sample_weight=w))
                         if len(np.unique(y[:, k])) == 2 else None for k in range(c)]
        defined = [v for v in per_label_auc if v is not None]
        auc = float(np.mean(defined)) if defined else None
        reason = "Macro AUROC excludes labels without both outcomes" if len(defined) < c else None
    return {"status": "ok", "n": int(len(y)), "weight": float(w.sum()),
            "accuracy": float(accuracy), "macro_f1": float(f1), "uar": float(recall),
            "auroc": None if auc is None else float(auc), "auroc_note": reason,
            "per_label_auroc": per_label_auc, "nll": float(nll), "ece": ece}
