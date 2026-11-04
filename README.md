# ObsAlign

**No Free Alignment: Observability-Aware Alignment for Multimodal Heterogeneous Learning**.

ObsAlign estimates whether a modality pair is supported for each semantic anchor.
The resulting scores control alignment, missing-modality calibration and aggregation
across heterogeneous data sources. Source-private residuals remain local.

## Observability score

The score in `obsalign/observability.py` follows Eq. (6) of the paper:

$$
O_{m,n,s} = \left(1-e^{-\mathrm{Bridge}_{m,n,s}}\right)
\sigma\!\left(\alpha\,\mathrm{Bridge}_{m,n,s}
+\beta\,\mathrm{Cons}_{m,n,s}+\eta\,\mathrm{Supp}_{m,n,s}
-\delta\,\mathrm{Unc}_{m,n,s}-b_0\right).
$$

Here $\mathrm{Bridge}_{m,n,s}=\log(1+D_{m,n,s}+\gamma_{\mathrm{path}}P_{m,n,s})$,
where $D$ is direct bridge evidence and $P$ is multi-hop path evidence. The bridge
sufficiency factor is equivalently
$(D+\gamma_{\mathrm{path}}P)/(1+D+\gamma_{\mathrm{path}}P)$.
Thus, a relation with neither direct nor multi-hop bridge evidence has zero
observability and zero alignment gate, even when both modalities have side support.
Path-only evidence can give positive observability when `gamma_path > 0`.

The implementation uses `-torch.expm1(-bridge)` to evaluate the factor accurately
for small bridge values. Defaults are
$(\alpha,\beta,\eta,\delta,b_0)=(1,1,0.5,1,0)$.
Run the equation regression checks from the repository root with:

```bash
python -m unittest discover -s tests -v
```

## Quick start

Python 3.9+ is required. Run these commands from the repository root:

```bash
python -m pip install .
python scripts/make_demo.py --output data/demo
python -m obsalign train --data data/demo --config configs/demo.yaml --output runs/demo
python -m obsalign evaluate --data data/demo --checkpoint runs/demo/best.pt
```

## Repository

```text
obsalign/
  model.py          Shared encoders, projections and private residuals
  sketch.py         Semantic prototypes, variance and reliability
  observability.py  Bridges, multi-hop paths and transport consistency
  losses.py         Task, alignment and calibration objectives
  aggregation.py    Evidence-aware shared-model aggregation
  trainer.py        Training rounds, checkpoints and resume
  data.py           Dataset loading and validation
  evaluation.py     Evaluation and unlabeled prediction
  metrics.py        Classification and calibration metrics
  config.py         Parameters and configuration validation
  cli.py            Command-line entry points
```

## Training on your data

Prepare a directory following [the dataset format](docs/data.md), then pass that
directory with `--data`. All sources share the modality order and label space;
their observation masks and semantic coverage may differ.

Configuration is YAML. Defaults are listed in [configs/default.yaml](configs/default.yaml).
Repeat `--set` to override individual settings. For example, a larger demo run is:

```bash
python -m obsalign train --data data/demo --config configs/default.yaml --output runs/demo-long --set rounds=100
```

Use `--set device=cuda` for an available CUDA device. The supplied MLP encoders
accept feature vectors or flattened signal tensors. Replace the encoders in
`model.py` when a benchmark requires a specialized backbone.

Each run writes `config.json`, `history.jsonl`, `latest.pt`, `best.pt` and
`observation.npz`. The best checkpoint is selected on validation macro-F1 by default.
The test split is evaluated only by the separate evaluation command.

```bash
python -m obsalign train --data data/demo --output runs/demo --resume runs/demo/latest.pt --set rounds=20
```

Resume uses the saved configuration and requires the same data and output directory.
The total number of rounds, device and CPU thread count may change.

## Evaluation and prediction

```bash
python -m obsalign evaluate --data data/demo --checkpoint runs/demo/best.pt --output runs/demo/test_metrics.json
python -m obsalign predict --inputs data/demo/inputs.npz --checkpoint runs/demo/best.pt --output runs/demo/predictions.npz
```

Add `--personalize` to use saved private branches for known source IDs. Shared
prediction is the default. Metrics include accuracy, macro-F1, UAR, AUROC, NLL and
ECE; undefined AUROC values are returned as `null` with an explanatory field.

