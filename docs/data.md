# Dataset format

A dataset directory contains `metadata.json`, `train.npz`, `val.npz` and `test.npz`.
The executable writer in [make_demo.py](../scripts/make_demo.py) illustrates the format.

## Metadata

```json
{
  "modalities": ["image", "audio", "text"],
  "num_classes": 3,
  "task": "multiclass"
}
```

Use `task: "multilabel"` for binary label vectors. An optional `num_semantics`
sets the semantic-anchor count; it defaults to `num_classes`.
Modality names, their order, class IDs and feature shapes must agree across splits.

## Arrays in each NPZ

| Key | Shape | Meaning |
| --- | --- | --- |
| `x_image`, `x_audio`, `x_text` | `[N, D_m]` or `[N, channels, time]` | Float32 features; one array per named modality |
| `y` | `[N]` or `[N, C]` | Integer class IDs starting at 0, or binary multilabel targets |
| `source` | `[N]` | Nonnegative integer source IDs |
| `observed` | `[N, M]` | Boolean visibility mask in metadata modality order |
| `ids` | `[N]` | Optional unique string sample IDs, shared across all modalities |
| `semantic` | `[N, S]` | Optional nonnegative semantic assignments whose rows sum to one |

Every sample needs at least one visible modality. Missing feature values are
ignored and replaced by zeros by the loader. Supply an array for every modality,
including wholly missing ones. Source-level absence is inferred from the source's
training masks, so the same format supports both source and sample missingness.

Without `semantic`, multiclass labels become one-hot anchors and multilabel labels
become normalized multi-hot anchors. Multilabel rows with no positive labels need
explicit semantic assignments. Classifiers still use the original target labels.

Split samples before feature extraction or preprocessing that learns statistics.
Fit such preprocessing on training data only. Use real sample IDs to let the
loader detect overlap across splits; auto-generated IDs cannot detect duplicate
records supplied under different files. No benchmark-specific partition is inferred.

## Unlabeled prediction

`inputs.npz` contains the same feature arrays and `observed`. Labels and semantic
assignments are unnecessary. `source` and `ids` are optional. With `--personalize`,
known source IDs use their saved private branches; unseen or omitted IDs use the
shared model. Output contains `ids`, `probabilities`, `predictions` and `observed`.
