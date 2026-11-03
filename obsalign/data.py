"""Validated NumPy bundles with explicit source IDs and observation masks."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import torch


@dataclass
class Split:
    x: dict
    y: object
    source: torch.Tensor
    observed: torch.Tensor
    semantic: object
    ids: np.ndarray

    def __len__(self):
        return len(self.observed)

    def subset(self, index):
        index = torch.as_tensor(index, dtype=torch.long)
        return Split({m: x[index] for m, x in self.x.items()},
                     None if self.y is None else self.y[index], self.source[index], self.observed[index],
                     None if self.semantic is None else self.semantic[index], self.ids[index.numpy()])


def validate_metadata(meta):
    modalities = meta["modalities"]
    if len(modalities) < 2 or len(set(modalities)) != len(modalities):
        raise ValueError("Provide at least two unique modality names")
    if any(not name.isidentifier() for name in modalities):
        raise ValueError("Modality names must be valid identifiers")
    if meta["task"] not in {"multiclass", "multilabel"} or int(meta["num_classes"]) < 2:
        raise ValueError("Use multiclass or multilabel with at least two classes")
    if int(meta.get("num_semantics", meta["num_classes"])) < 1:
        raise ValueError("num_semantics must be positive")


def load_split(path, meta, labeled=True, shapes=None):
    modalities, classes = meta["modalities"], int(meta["num_classes"])
    semantics = int(meta.get("num_semantics", classes))
    with np.load(path, allow_pickle=False) as arrays:
        raw_mask = arrays["observed"]
        if raw_mask.ndim != 2 or raw_mask.shape[1] != len(modalities) or not np.isin(raw_mask, [0, 1]).all():
            raise ValueError("observed must be a binary [N, M] array")
        mask = torch.from_numpy(raw_mask.copy()).bool()
        n = len(mask)
        if n == 0 or not mask.any(1).all():
            raise ValueError("Every split and sample must have observed data")
        raw_source = arrays["source"] if "source" in arrays else np.full(n, -1, dtype=np.int64)
        if raw_source.shape != (n,) or raw_source.dtype.kind not in "iu" or (labeled and (raw_source < 0).any()):
            raise ValueError("Labeled data require nonnegative integer source IDs, shape [N]")
        x = {}
        for column, modality in enumerate(modalities):
            value = torch.from_numpy(np.array(arrays[f"x_{modality}"], dtype=np.float32))
            if value.ndim < 2 or len(value) != n or any(d < 1 for d in value.shape[1:]):
                raise ValueError(f"Invalid input shape for {modality}")
            if shapes is not None and tuple(value.shape[1:]) != tuple(shapes[modality]):
                raise ValueError(f"Feature shape differs from training for {modality}")
            if not torch.isfinite(value[mask[:, column]]).all():
                raise ValueError(f"Observed inputs must be finite: {modality}")
            value[~mask[:, column]] = 0
            x[modality] = value
        y, assignment = None, None
        if labeled:
            y = torch.from_numpy(arrays["y"].copy())
            if meta["task"] == "multiclass":
                if y.shape != (n,) or y.is_floating_point() or (y < 0).any() or (y >= classes).any():
                    raise ValueError("Multiclass labels must be integer [N] in [0, num_classes)")
                y = y.long()
                default = torch.nn.functional.one_hot(y, classes).float()
            else:
                if y.shape != (n, classes) or not ((y == 0) | (y == 1)).all():
                    raise ValueError("Multilabel targets must be binary [N, num_classes]")
                y = y.float()
                default = y / y.sum(1, keepdim=True).clamp_min(1)
            assignment = torch.from_numpy(arrays["semantic"].copy()).float() if "semantic" in arrays else default
            if assignment.shape != (n, semantics) or not torch.isfinite(assignment).all() or (assignment < 0).any():
                raise ValueError("semantic must be a finite nonnegative [N, num_semantics] array")
            if not torch.allclose(assignment.sum(1), torch.ones(n), atol=1e-5):
                raise ValueError("Semantic assignments must sum to one; zero-positive labels need explicit semantic")
        ids = arrays["ids"].astype(str) if "ids" in arrays else np.array([f"{Path(path).stem}:{i}" for i in range(n)])
        if ids.shape != (n,) or len(set(ids)) != n:
            raise ValueError("Sample IDs must be unique within a split")
    return Split(x, y, torch.from_numpy(raw_source.copy()).long(), mask, assignment, ids)


class Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        validate_metadata(self.meta)
        self.modalities = self.meta["modalities"]
        self.classes, self.task = int(self.meta["num_classes"]), self.meta["task"]
        train = load_split(self.root / "train.npz", self.meta)
        self.shapes = {m: tuple(v.shape[1:]) for m, v in train.x.items()}
        self.splits = {"train": train}
        self.splits.update({name: load_split(self.root / f"{name}.npz", self.meta, shapes=self.shapes)
                            for name in ("val", "test")})
        seen = set()
        for split in self.splits.values():
            if seen.intersection(split.ids):
                raise ValueError("Sample IDs overlap across train, val or test")
            seen.update(split.ids)

    def fingerprint(self):
        digest = hashlib.sha256()
        for name in ("metadata.json", "train.npz", "val.npz", "test.npz"):
            digest.update(name.encode())
            with (self.root / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
