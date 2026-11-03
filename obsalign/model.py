"""Shared encoders and projections; source-private residuals stay local."""
import math
import torch
from torch import nn


def normalize(x, eps=1e-8):
    return x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + eps)


class SharedModel(nn.Module):
    def __init__(self, shapes, classes, cfg):
        super().__init__()
        self.modalities, self.cfg = list(shapes), cfg
        self.encoders, self.projections = nn.ModuleDict(), nn.ModuleDict()
        for modality, shape in shapes.items():
            self.encoders[modality] = nn.Sequential(
                nn.Flatten(), nn.Linear(math.prod(shape), cfg.hidden_dim), nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU())
            self.projections[modality] = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.embedding_dim))
        self.head = nn.Linear(cfg.embedding_dim, classes)

    def representations(self, x, mask, private=None):
        shared, adapted = [], []
        for column, modality in enumerate(self.modalities):
            rows = torch.where(mask[:, column])[0]
            empty = x[modality].new_zeros(len(mask), self.cfg.embedding_dim)
            z, personalized = empty, empty
            if len(rows):
                inputs = x[modality][rows]
                h = self.encoders[modality](inputs)
                value = normalize(self.projections[modality](h), self.cfg.eps)
                z = empty.index_copy(0, rows, value)
                if private is not None:
                    value = normalize(self.projections[modality](h + private.branches[modality](inputs)), self.cfg.eps)
                personalized = empty.index_copy(0, rows, value)
            shared.append(z)
            adapted.append(personalized)
        return torch.stack(shared, 1), torch.stack(adapted, 1)

    def forward(self, x, mask, private=None):
        if not mask.any(1).all():
            raise ValueError("Every sample needs an observed modality")
        shared, adapted = self.representations(x, mask, private)
        fused = adapted.sum(1) / mask.sum(1, keepdim=True).to(adapted.dtype)
        return self.head(fused), shared


class PrivateResidual(nn.Module):
    def __init__(self, shapes, cfg):
        super().__init__()
        self.branches = nn.ModuleDict()
        for modality, shape in shapes.items():
            branch = nn.Sequential(nn.Flatten(), nn.Linear(math.prod(shape), cfg.residual_width),
                                   nn.GELU(), nn.Linear(cfg.residual_width, cfg.hidden_dim))
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
            self.branches[modality] = branch

    def penalty(self, visible):
        terms = [p.square().sum() for modality, use in zip(self.branches, visible) if bool(use)
                 for p in self.branches[modality].parameters()]
        return torch.stack(terms).sum()
