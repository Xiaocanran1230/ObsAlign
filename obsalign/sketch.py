"""Equations (3), (4), (12), (13); source sketches never use private features."""
from dataclasses import dataclass
import torch


@dataclass
class Sketch:
    count: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    reliability: torch.Tensor
    visible: torch.Tensor

    def state_dict(self):
        return {k: getattr(self, k).detach().cpu() for k in self.__dataclass_fields__}

    @classmethod
    def from_state(cls, state):
        return cls(**state)

    def evidence(self, cfg):
        return (self.visible[:, None] & (self.count > 0) & (self.count >= cfg.n_min)
                & (self.reliability >= cfg.rho_min))

    def payload_bytes(self):
        """Eq. (79), float32 wire arrays, excluding headers/visible mask."""
        return 4 * int(self.visible.sum()) * self.count.shape[1] * (self.mean.shape[-1] + 3)


def centers(z, assignment, mask, eps=1e-8):
    """Differentiable centers: [B,M,D], [B,S], [B,M] -> [M,S,D], [M,S]."""
    weights = assignment[:, None, :] * mask[:, :, None].to(z.dtype)
    count = weights.sum(0)
    mean = torch.einsum("bms,bmd->msd", weights, z) / (count[..., None] + eps)
    return mean, count


def sketch_from_moments(count, total, square_total, visible, cfg):
    mean = total / (count[..., None] + cfg.eps)
    # Exact algebra for sum a ||z - mu||^2, including the epsilon in the mean.
    var = (square_total - 2 * (mean * total).sum(-1) + count * mean.square().sum(-1))
    var = (var / (count + cfg.eps)).clamp_min(0)
    rho = count / (count + cfg.n0) * torch.exp(-var / cfg.tau_v)
    return Sketch(count, mean, var, rho, visible)


def make_sketch(z, assignment, mask, cfg):
    w = assignment[:, None, :] * mask[:, :, None].to(z.dtype)
    return sketch_from_moments(w.sum(0), torch.einsum("bms,bmd->msd", w, z),
                               torch.einsum("bms,bm->ms", w, z.square().sum(-1)), mask.any(0), cfg)


@torch.no_grad()
def collect_sketch(model, split, mask, cfg):
    model.eval()
    device = next(model.parameters()).device
    m, s, d = len(model.modalities), split.semantic.shape[1], cfg.embedding_dim
    # Float64 accumulation on CPU limits roundoff without retaining a computation graph.
    count = torch.zeros(m, s, dtype=torch.float64)
    total = torch.zeros(m, s, d, dtype=torch.float64)
    squares = torch.zeros(m, s, dtype=torch.float64)
    for start in range(0, len(split), cfg.batch_size):
        sl = slice(start, start + cfg.batch_size)
        x = {k: v[sl].to(device) for k, v in split.x.items()}
        visible = mask[sl].to(device)
        z, _ = model.representations(x, visible, private=None)
        z = z.cpu().double()
        w = split.semantic[sl, None, :].double() * mask[sl, :, None].double()
        count += w.sum(0)
        total += torch.einsum("bms,bmd->msd", w, z)
        squares += torch.einsum("bms,bm->ms", w, z.square().sum(-1))
    result = sketch_from_moments(count, total, squares, mask.any(0), cfg)
    return Sketch(result.count.float(), result.mean.float(), result.variance.float(),
                  result.reliability.float(), result.visible)
