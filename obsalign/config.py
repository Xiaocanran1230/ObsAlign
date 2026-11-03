"""Training and observability parameters, loaded from YAML or JSON."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json
import math
import yaml


@dataclass
class Config:
    seed: int = 0
    rounds: int = 200
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-5
    server_lr: float = 1.0
    hidden_dim: int = 256
    embedding_dim: int = 128
    residual_width: int = 64
    lambda_align: float = 0.1
    lambda_cal: float = 0.01
    lambda_residual: float = 1e-4
    n0: float = 10.0
    n_min: float = 1.0
    rho_min: float = 0.05
    tau_v: float = 1.0
    tau_b: float = 1.0
    tau_c: float = 1.0
    tau_q: float = 0.1
    lambda_transport: float = 1.0
    d_min: float = 1e-4
    max_path_length: int = 3
    gamma0: float = 0.5
    gamma_path: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    eta: float = 0.5
    delta: float = 1.0
    b0: float = 0.0
    tau_low: float = 0.2
    tau_high: float = 0.6
    zeta: float = 0.5
    eps: float = 1e-8
    prototype_scope: str = "batch"
    primary_metric: str = "macro_f1"
    device: str = "cpu"
    threads: int = 1

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        for name in ("rounds", "local_epochs", "batch_size", "hidden_dim", "embedding_dim",
                     "residual_width", "max_path_length", "threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("lr", "n0", "tau_v", "tau_b", "tau_c", "tau_q", "lambda_transport", "eps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("weight_decay", "server_lr", "lambda_align", "lambda_cal", "lambda_residual",
                     "n_min", "d_min", "gamma_path", "alpha", "beta", "eta", "delta", "zeta"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.tau_low < self.tau_high <= 1:
            raise ValueError("Require 0 <= tau_low < tau_high <= 1")
        if not 0 <= self.rho_min <= 1 or not 0 < self.gamma0 < 1 or not 0 <= self.momentum < 1:
            raise ValueError("Invalid reliability threshold, path decay or momentum")
        if self.prototype_scope not in {"batch", "source"}:
            raise ValueError("prototype_scope must be batch or source")
        if self.primary_metric not in {"accuracy", "macro_f1", "uar", "auroc"}:
            raise ValueError("Unknown primary_metric")
        return self

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values).validate()

    @classmethod
    def load(cls, path=None, overrides=()):
        values = {}
        if path is not None:
            path = Path(path)
            text = path.read_text(encoding="utf-8")
            values = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        if not isinstance(values, dict):
            raise ValueError("Config must be a mapping")
        for override in overrides:
            key, separator, value = override.partition("=")
            if not separator:
                raise ValueError("Overrides must have the form key=value")
            try:
                values[key] = json.loads(value)
            except json.JSONDecodeError:
                values[key] = yaml.safe_load(value)
        return cls.from_dict(values)

    def save(self, path):
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
