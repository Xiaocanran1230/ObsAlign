"""Eq. (10): consolidate shared parameters with bridge-aware source weights."""
import torch
from .observability import source_bridge


def consolidation_weights(sketches, sample_counts, old_gate, cfg):
    pairs = torch.triu_indices(old_gate.shape[0], old_gate.shape[1], 1)
    chi = torch.stack([(old_gate * source_bridge(s, cfg))[pairs[0], pairs[1]].sum() for s in sketches])
    weights = torch.as_tensor(sample_counts, dtype=chi.dtype) * (1 + cfg.zeta * chi)
    weights = weights / (weights.sum() + cfg.eps)
    return weights, chi


def aggregate(base, local_states, sketches, sample_counts, old_gate, cfg):
    # O^t is held fixed here. The new sketches determine bridge strengths only.
    weights, chi = consolidation_weights(sketches, sample_counts, old_gate, cfg)
    state = {key: value + cfg.server_lr * sum(
        weight * (local[key] - value) for weight, local in zip(weights, local_states))
        for key, value in base.items()}
    return state, {"weights": weights.tolist(), "bridge_credit": chi.tolist()}
