"""Closed-form transport and exact observability components, Eqs. (5)-(7), (14)-(24)."""
from dataclasses import dataclass
import torch


@dataclass
class Observation:
    prototypes: torch.Tensor
    support: torch.Tensor
    side_uncertainty: torch.Tensor
    direct: torch.Tensor
    paths: torch.Tensor
    bridge: torch.Tensor
    consistency: torch.Tensor
    side_support: torch.Tensor
    uncertainty: torch.Tensor
    score: torch.Tensor
    gate: torch.Tensor
    transport: torch.Tensor

    def state_dict(self):
        return {k: getattr(self, k).detach().cpu() for k in self.__dataclass_fields__}

    def to(self, device):
        return Observation(**{k: v.to(device) for k, v in self.state_dict().items()})

    @classmethod
    def from_state(cls, state):
        return cls(**state)


def ridge_transport(x, y, weights, regularization):
    """Eq. (21), row samples x/y; returned T acts on column vectors.

    T = I + (Y-X)^T W X (X^T W X + lambda I)^-1.
    Solve the equivalent K-by-K system when K < d (Woodbury identity).
    """
    if regularization <= 0:
        raise ValueError("Transport regularization must be positive")
    d = x.shape[-1]
    identity = torch.eye(d, dtype=x.dtype, device=x.device)
    keep = weights > 0
    x, y, weights = x[keep], y[keep], weights[keep]
    if len(x) == 0:
        return identity
    w = weights.sqrt()[:, None]
    a, difference = x * w, (y - x) * w
    if len(x) < d:
        lhs = a @ a.T + regularization * torch.eye(len(x), dtype=x.dtype, device=x.device)
        return identity + difference.T @ torch.linalg.solve(lhs, a)
    lhs = a.T @ a + regularization * identity
    rhs = (y * w).T @ a + regularization * identity
    return torch.linalg.solve(lhs, rhs.T).T


def source_bridge(sketch, cfg):
    """Eq. (16); zero diagonal, symmetric [M,M,S]."""
    e = sketch.evidence(cfg).to(sketch.mean.dtype)
    r = e * sketch.reliability
    distance = (sketch.mean[:, None] - sketch.mean[None, :]).square().sum(-1)
    b = r[:, None] * r[None, :] * torch.exp(-distance / cfg.tau_b)
    offdiag = ~torch.eye(len(r), dtype=torch.bool, device=r.device)
    return b * offdiag[:, :, None]


def relation_gate(score, low, high):
    return score * ((score - low) / (high - low)).clamp(0, 1)




@torch.no_grad()
def estimate(sketches, cfg):
    if not sketches:
        raise ValueError("At least one sketch is required")
    means = torch.stack([z.mean for z in sketches]).double().cpu()
    counts = torch.stack([z.count for z in sketches]).double().cpu()
    rho = torch.stack([z.reliability for z in sketches]).double().cpu()
    variance = torch.stack([z.variance for z in sketches]).double().cpu()
    evidence = torch.stack([z.evidence(cfg) for z in sketches]).double().cpu()
    weights = evidence * counts * rho
    support = weights.sum(0)
    prototypes = (weights[..., None] * means).sum(0) / (support[..., None] + cfg.eps)
    side_uncertainty = (weights * (variance + (means - prototypes).square().sum(-1))).sum(0)
    side_uncertainty /= support + cfg.eps
    bridges = torch.stack([source_bridge(z, cfg) for z in sketches]).double().cpu()
    direct = bridges.sum(0)
    m, _, s = direct.shape
    d = means.shape[-1]
    adjacency = direct.permute(2, 0, 1)
    degree = adjacency.sum(-1)
    inv_sqrt = torch.zeros_like(degree)
    positive = degree > 0
    inv_sqrt[positive] = degree[positive].rsqrt()
    normalized = inv_sqrt[:, :, None] * adjacency * inv_sqrt[:, None, :]
    power = normalized.clone()
    paths = torch.zeros_like(normalized)
    for length in range(2, cfg.max_path_length + 1):
        power = power @ normalized
        paths += cfg.gamma0 ** (length - 1) * power
    paths = paths.permute(1, 2, 0)
    bridge = torch.log1p(direct + cfg.gamma_path * paths)
    side_support = torch.minimum(torch.log1p(support)[:, None], torch.log1p(support)[None, :])
    uncertainty = (side_uncertainty[:, None] + side_uncertainty[None, :]) / 2 + 1 / (1 + direct)
    transport = torch.eye(d, dtype=torch.float64).expand(m, m, s, d, d).clone()
    consistency = torch.zeros_like(direct)
    for first in range(m):
        for second in range(first + 1, m):
            for semantic in range(s):
                w = bridges[:, first, second, semantic]
                x, y = means[:, first, semantic], means[:, second, semantic]
                forward = ridge_transport(x, y, w, cfg.lambda_transport)
                backward = ridge_transport(y, x, w, cfg.lambda_transport)
                transport[first, second, semantic] = forward
                transport[second, first, semantic] = backward
                if direct[first, second, semantic] >= cfg.d_min:
                    a, b = prototypes[first, semantic], prototypes[second, semantic]
                    error = (a - backward @ forward @ a).square().sum()
                    error += (b - forward @ backward @ b).square().sum()
                    value = torch.exp(-error / (2 * cfg.tau_c))
                    consistency[first, second, semantic] = value
                    consistency[second, first, semantic] = value
    # Eq. (6): expm1 preserves small positive bridge evidence in 1 - exp(-bridge).
    bridge_sufficiency = -torch.expm1(-bridge)
    score = bridge_sufficiency * torch.sigmoid(
        cfg.alpha * bridge + cfg.beta * consistency + cfg.eta * side_support
        - cfg.delta * uncertainty - cfg.b0)
    offdiag = (~torch.eye(m, dtype=torch.bool))[:, :, None]
    score *= offdiag
    gate = relation_gate(score, cfg.tau_low, cfg.tau_high)
    values = dict(prototypes=prototypes, support=support, side_uncertainty=side_uncertainty,
                  direct=direct, paths=paths, bridge=bridge, consistency=consistency,
                  side_support=side_support, uncertainty=uncertainty, score=score, gate=gate,
                  transport=transport)
    return Observation(**{k: v.float().detach() for k, v in values.items()})
