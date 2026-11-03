"""Shared and personalized prediction, including unlabeled NPZ inputs."""
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .data import load_split
from .metrics import classification_metrics
from .model import SharedModel, PrivateResidual


@torch.no_grad()
def predict(model, split, shapes, task, cfg, private_states=None):
    model.eval()
    device = next(model.parameters()).device
    probabilities = torch.empty(len(split), model.head.out_features)
    for source in torch.unique(split.source).tolist():
        rows = torch.where(split.source == source)[0]
        private = None
        if private_states is not None and source in private_states:
            private = PrivateResidual(shapes, cfg).to(device).eval()
            private.load_state_dict(private_states[source])
        for index in rows.split(cfg.batch_size):
            x = {m: value[index].to(device) for m, value in split.x.items()}
            logits, _ = model(x, split.observed[index].to(device), private)
            p = logits.softmax(-1) if task == "multiclass" else logits.sigmoid()
            probabilities[index] = p.cpu()
    return probabilities.numpy()


def load_model(checkpoint, device="cpu"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    cfg = Config.from_dict({**state["config"], "device": device})
    torch.set_num_threads(cfg.threads)
    model = SharedModel(state["shapes"], state["metadata"]["num_classes"], cfg).to(device).eval()
    model.load_state_dict(state["model"])
    return model, cfg, state


def evaluate_checkpoint(data, checkpoint, split="test", device="cpu", personalize=False):
    model, cfg, state = load_model(checkpoint, device)
    values = load_split(Path(data) / f"{split}.npz", state["metadata"], shapes=state["shapes"])
    p = predict(model, values, state["shapes"], state["metadata"]["task"], cfg,
                state["private"] if personalize else None)
    result = classification_metrics(values.y.numpy(), p, state["metadata"]["task"])
    result.update(checkpoint_round=state["round"], split=split, personalized=personalize)
    return result


def infer(checkpoint, inputs, output, device="cpu", personalize=False):
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("Prediction output must end in .npz")
    model, cfg, state = load_model(checkpoint, device)
    values = load_split(inputs, state["metadata"], labeled=False, shapes=state["shapes"])
    task = state["metadata"]["task"]
    p = predict(model, values, state["shapes"], task, cfg, state["private"] if personalize else None)
    labels = p.argmax(1) if task == "multiclass" else (p >= 0.5).astype(np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=values.ids, probabilities=p, predictions=labels,
                        observed=values.observed.numpy())
    return {"samples": len(values), "output": str(output), "checkpoint_round": state["round"]}
