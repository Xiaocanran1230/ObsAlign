"""Algorithm 1 with persistent private branches and resumable shared training."""
from pathlib import Path
import copy
import json
import time
import numpy as np
import torch
from .aggregation import aggregate
from .evaluation import predict
from .losses import task_loss, relation_losses
from .metrics import classification_metrics
from .model import SharedModel, PrivateResidual
from .observability import Observation, estimate
from .sketch import centers, collect_sketch


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_checkpoint(path, state):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


class Trainer:
    def __init__(self, data, cfg, output):
        self.data, self.cfg, self.output = data, cfg.validate(), Path(output)
        torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.model = SharedModel(data.shapes, data.classes, cfg).to(self.device)
        self.private_states = {}
        self.observation = None
        self.round, self.best_round, self.best_score = 0, 0, -float("inf")
        self.data_hash = data.fingerprint()
        train = data.splits["train"]
        self.indices = {int(s): torch.where(train.source == s)[0] for s in torch.unique(train.source)}

    def make_private(self, source):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.cfg.seed + 100003 + source)
            private = PrivateResidual(self.data.shapes, self.cfg)
        if source in self.private_states:
            private.load_state_dict(self.private_states[source])
        return private.to(self.device)

    def source_data(self, source):
        return self.data.splits["train"].subset(self.indices[source])

    def warm_start(self):
        sketches = []
        for source in self.indices:
            split = self.source_data(source)
            sketches.append(collect_sketch(self.model, split, split.observed, self.cfg))
        self.observation = estimate(sketches, self.cfg)

    def source_centers(self, model, split):
        shared = []
        for index in torch.arange(len(split)).split(self.cfg.batch_size):
            x = {m: value[index].to(self.device) for m, value in split.x.items()}
            z, _ = model.representations(x, split.observed[index].to(self.device))
            shared.append(z)
        return centers(torch.cat(shared), split.semantic.to(self.device),
                       split.observed.to(self.device), self.cfg.eps)

    def local_train(self, source):
        cfg, split = self.cfg, self.source_data(source)
        local, private = copy.deepcopy(self.model).train(), self.make_private(source).train()
        parameters = list(local.parameters()) + list(private.parameters())
        optimizer = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        observation = self.observation.to(self.device)
        visible = split.observed.any(0).to(self.device)
        generator = torch.Generator().manual_seed(cfg.seed + 1000003 * self.round + 997 * source)
        totals = {key: 0.0 for key in ("task", "align", "calibration", "residual", "total")}
        examples = 0
        for _ in range(cfg.local_epochs):
            for index in torch.randperm(len(split), generator=generator).split(cfg.batch_size):
                x = {m: value[index].to(self.device) for m, value in split.x.items()}
                mask = split.observed[index].to(self.device)
                logits, shared = local(x, mask, private)
                task = task_loss(logits, split.y[index].to(self.device), self.data.task)
                if cfg.prototype_scope == "source":
                    mean, count = self.source_centers(local, split)
                else:
                    mean, count = centers(shared, split.semantic[index].to(self.device), mask, cfg.eps)
                relations = relation_losses(mean, count, visible, observation, cfg)
                residual = private.penalty(visible)
                loss = (task + cfg.lambda_align * relations["align"] + cfg.lambda_cal * relations["calibration"]
                        + cfg.lambda_residual * residual)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at round {self.round}, source {source}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError(f"Nonfinite gradient at round {self.round}, source {source}")
                optimizer.step()
                for key, value in dict(task=task, **relations, residual=residual, total=loss).items():
                    totals[key] += float(value.detach()) * len(index)
                examples += len(index)
        self.private_states[source] = cpu_state(private)
        sketch = collect_sketch(local, split, split.observed, cfg)
        return cpu_state(local), sketch, {key: value / examples for key, value in totals.items()}

    def checkpoint_state(self):
        return {"format_version": 1, "round": self.round, "config": self.cfg.as_dict(),
                "metadata": self.data.meta, "shapes": self.data.shapes, "data_sha256": self.data_hash,
                "model": cpu_state(self.model), "private": self.private_states,
                "observation": self.observation.state_dict(), "best_score": self.best_score,
                "best_round": self.best_round}

    def resume(self, checkpoint):
        checkpoint = Path(checkpoint)
        if checkpoint.name != "latest.pt" or checkpoint.resolve().parent != self.output.resolve():
            raise ValueError("Resume from latest.pt in the same output directory")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1 or state["data_sha256"] != self.data_hash:
            raise ValueError("Checkpoint format or dataset fingerprint mismatch")
        allowed = {"rounds", "device", "threads"}
        changed = [key for key, value in self.cfg.as_dict().items()
                   if key not in allowed and value != state["config"][key]]
        if changed or self.cfg.rounds < state["round"]:
            raise ValueError(f"Resume must retain the training settings and completed rounds: {changed}")
        best = torch.load(self.output / "best.pt", map_location="cpu", weights_only=True)
        if best["data_sha256"] != self.data_hash or best["round"] != state["best_round"]:
            raise ValueError("best.pt does not match the resumed run")
        self.model.load_state_dict(state["model"])
        self.private_states = state["private"]
        self.observation = Observation.from_state(state["observation"])
        self.round, self.best_round, self.best_score = state["round"], state["best_round"], state["best_score"]
        history = self.output / "history.jsonl"
        if history.exists():
            records = [json.loads(line) for line in history.read_text().splitlines()]
            history.write_text("".join(json.dumps(r) + "\n" for r in records if r["round"] <= self.round))

    def evaluate(self, split="val", personalize=False):
        values = self.data.splits[split]
        p = predict(self.model, values, self.data.shapes, self.data.task, self.cfg,
                    self.private_states if personalize else None)
        return classification_metrics(values.y.numpy(), p, self.data.task)

    def fit(self, resume=None):
        self.output.mkdir(parents=True, exist_ok=True)
        if resume:
            self.resume(resume)
        elif any((self.output / name).exists() for name in ("latest.pt", "history.jsonl")):
            raise FileExistsError("Run already exists; use --resume or another output directory")
        else:
            self.warm_start()
        self.cfg.save(self.output / "config.json")
        while self.round < self.cfg.rounds:
            start = time.perf_counter()
            states, sketches, losses = [], [], []
            for source in self.indices:
                state, sketch, loss = self.local_train(source)
                states.append(state)
                sketches.append(sketch)
                losses.append(loss)
            # Consolidate using O^t and fresh local sketches, then construct O^(t+1).
            state, weights = aggregate(cpu_state(self.model), states, sketches,
                                       [len(index) for index in self.indices.values()],
                                       self.observation.gate, self.cfg)
            self.model.load_state_dict(state)
            self.observation = estimate(sketches, self.cfg)
            self.round += 1
            metrics = self.evaluate("val")
            score = metrics.get(self.cfg.primary_metric)
            if score is None:
                raise ValueError(f"Validation metric {self.cfg.primary_metric} is undefined: {metrics}")
            improved = score > self.best_score
            if improved:
                self.best_score, self.best_round = score, self.round
            record = dict(round=self.round, sources=list(self.indices), losses=losses, aggregation=weights,
                          validation=metrics, seconds=time.perf_counter() - start)
            with (self.output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            checkpoint = self.checkpoint_state()
            if improved:
                save_checkpoint(self.output / "best.pt", checkpoint)
            save_checkpoint(self.output / "latest.pt", checkpoint)
            print(f"round={self.round:03d} val_{self.cfg.primary_metric}={score:.4f} "
                  f"best={self.best_score:.4f}", flush=True)
        np.savez_compressed(self.output / "observation.npz", **{
            key: value.numpy() for key, value in self.observation.state_dict().items() if key != "transport"})
        return {"rounds": self.round, "best_round": self.best_round, "best_validation": self.best_score,
                "metric": self.cfg.primary_metric, "checkpoint": str(self.output / "best.pt")}
