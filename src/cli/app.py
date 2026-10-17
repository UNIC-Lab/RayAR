from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

from src.config import AppConfig, DataConfig, load_config


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _config(args) -> AppConfig:
    base = load_config(args.config)
    data = DataConfig(
        **{
            **base.data.__dict__,
            "building_root": (
                Path(args.building_root)
                if args.building_root
                else base.data.building_root
            ),
            "los_root": (Path(args.los_root) if args.los_root else base.data.los_root),
            "cache_root": (
                Path(args.cache_root) if args.cache_root else base.data.cache_root
            ),
        }
    )
    train = replace(
        base.train,
        batch_size=(
            args.batch_size
            if getattr(args, "batch_size", None) is not None
            else base.train.batch_size
        ),
        num_workers=(
            args.num_workers
            if getattr(args, "num_workers", None) is not None
            else base.train.num_workers
        ),
        epochs=(
            args.epochs
            if getattr(args, "epochs", None) is not None
            else base.train.epochs
        ),
        lr=(args.lr if getattr(args, "lr", None) is not None else base.train.lr),
    )
    return AppConfig(
        data=data,
        model=base.model,
        train=train,
        experiment=base.experiment,
        overfit=base.overfit,
    )


def _device(args) -> str | None:
    gpu = getattr(args, "gpu", None)
    return f"cuda:{gpu}" if gpu is not None else getattr(args, "device", None)


def _add_device_args(command: argparse.ArgumentParser) -> None:
    group = command.add_mutually_exclusive_group()
    group.add_argument("--gpu", type=int)
    group.add_argument("--device")


def cmd_train(args) -> None:
    from src.train.runner import train

    path = train(
        _config(args),
        args.output,
        args.val_rx_limit,
        _device(args),
        args.init_checkpoint,
        args.resume,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(path)


def cmd_eval(args) -> None:
    from src.eval.runner import evaluate

    result = evaluate(
        _config(args),
        args.checkpoint,
        args.output,
        args.rx_limit,
        _device(args),
        args.map_id,
        args.sample_count,
        args.balanced_hops,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_infer(args) -> None:
    import numpy as np
    import torch

    from src.data.cache import load_map_cache
    from src.data.features import PAFeatureBuilder
    from src.eval.predict import AutoregressivePredictor
    from src.train.runner import load_checkpoint

    cfg = _config(args)
    device = torch.device(
        _device(args) or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache = load_map_cache(cfg.data.cache_root, args.map_id)
    builder = PAFeatureBuilder(
        cache,
        cfg.data.building_root,
        cfg.data.building_height,
        cfg.model.k_pre,
        cfg.model.max_wall_tokens,
        cfg.data.los_root,
    )
    model = load_checkpoint(args.checkpoint, device)
    predictor = AutoregressivePredictor(model, builder, device)
    tx = np.asarray(args.tx, dtype=np.float32)
    rx = np.asarray(args.rx, dtype=np.float32)
    greedy = predictor.greedy(tx, rx)
    print(
        json.dumps(
            {"greedy": greedy.__dict__},
            ensure_ascii=False,
        )
    )


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(description="IRT dominant-path prediction")
    app.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        help=argparse.SUPPRESS,
    )
    app.add_argument(
        "--config",
        default=str(PACKAGE_ROOT / "configs" / "full_100map_reference.json"),
    )
    app.add_argument("--building-root")
    app.add_argument("--los-root")
    app.add_argument("--cache-root")
    sub = app.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument(
        "--output",
        default=str(PACKAGE_ROOT / "output" / "full"),
    )
    train.add_argument("--val-rx-limit", type=int, default=200)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--lr", type=float)
    checkpoint = train.add_mutually_exclusive_group()
    checkpoint.add_argument("--init-checkpoint")
    checkpoint.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="resume from OUTPUT/last.pt, then OUTPUT/best.pt",
    )
    _add_device_args(train)
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument(
        "--output",
        default=str(PACKAGE_ROOT / "output" / "full" / "test_metrics.json"),
    )
    evaluate.add_argument("--rx-limit", type=int)
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--map-id", type=int)
    evaluate.add_argument("--sample-count", type=int)
    evaluate.add_argument("--balanced-hops", action="store_true")
    _add_device_args(evaluate)
    evaluate.set_defaults(func=cmd_eval)

    infer = sub.add_parser("infer")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--map-id", type=int, required=True)
    infer.add_argument(
        "--tx",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    infer.add_argument(
        "--rx",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    _add_device_args(infer)
    infer.set_defaults(func=cmd_infer)
    return app


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


__all__ = ["main", "parser"]
