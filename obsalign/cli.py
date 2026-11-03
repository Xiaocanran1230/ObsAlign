"""Command-line training, evaluation and unlabeled prediction."""
import argparse
import json
from pathlib import Path
from .config import Config
from .data import Dataset
from .evaluation import evaluate_checkpoint, infer
from .trainer import Trainer


def main():
    parser = argparse.ArgumentParser(prog="obsalign", description="Observability-aware multimodal alignment")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train on a source-partitioned NPZ dataset")
    train.add_argument("--data", required=True)
    train.add_argument("--config", default=None)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", default=None)
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    evaluate = commands.add_parser("evaluate", help="Evaluate a saved model")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate.add_argument("--output", default=None, help="Optional metrics JSON file")
    predict = commands.add_parser("predict", help="Predict from unlabeled NPZ inputs")
    predict.add_argument("--inputs", required=True)
    predict.add_argument("--output", required=True)
    for command in (evaluate, predict):
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--personalize", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        config = args.config
        if config is None and args.resume:
            config = str(Path(args.resume).parent / "config.json")
        trainer = Trainer(Dataset(args.data), Config.load(config, args.set), args.output)
        result = trainer.fit(args.resume)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(args.data, args.checkpoint, args.split, args.device, args.personalize)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        result = infer(args.checkpoint, args.inputs, args.output, args.device, args.personalize)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
