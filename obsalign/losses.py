"""Eqs. (8)-(9), (25)-(28): task supervision and relation objectives."""
import math
from torch.nn import functional as F
from .model import normalize


def task_loss(logits, y, task):
    return F.cross_entropy(logits, y) if task == "multiclass" else F.binary_cross_entropy_with_logits(logits, y)


def relation_losses(mean, count, visible, observation, cfg):
    align, calibration = mean.sum() * 0, mean.sum() * 0
    valid = count > 0
    modalities, semantics, _ = mean.shape
    for first in range(modalities):
        if not bool(visible[first]):
            continue
        for second in range(first + 1, modalities):
            if bool(visible[second]):
                distance = (mean[first] - mean[second]).square().sum(-1)
                align = align + (observation.gate[first, second] * distance * valid[first] * valid[second]).sum()
        for second in range(modalities):
            if bool(visible[second]):
                continue
            cosine = normalize(mean[first], cfg.eps) @ normalize(observation.prototypes[second], cfg.eps).T
            log_q = F.log_softmax(cosine / cfg.tau_q, dim=-1)
            kl = (log_q.exp() * (log_q + math.log(semantics))).sum(-1)
            score = observation.score[first, second]
            weight = (1 - score) * (score < cfg.tau_high) * valid[first]
            calibration = calibration + (weight * kl).sum()
    return {"align": align, "calibration": calibration}
