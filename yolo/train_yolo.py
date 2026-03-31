from argparse import ArgumentParser
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from ultralytics import YOLO
import yaml


def parse_args():
    parser = ArgumentParser(description="Train signpost YOLO detector")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="YOLO dataset yaml path, relative to this script by default",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base model checkpoint or model name understood by Ultralytics",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save-period", type=int, default=-1)

    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.5)
    parser.add_argument("--hsv-v", type=float, default=0.3)
    parser.add_argument("--degrees", type=float, default=10.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--shear", type=float, default=2.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    return parser.parse_args()


def resolve_from_script(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parent / candidate


def prepare_data_yaml(data_yaml: Path) -> Path:
    """
    Rewrite the dataset yaml with an absolute `path` so Ultralytics resolves
    train/val directories correctly regardless of the current working directory.
    """
    with data_yaml.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    base_path = data_cfg.get("path", ".")
    base_path = Path(base_path)
    if not base_path.is_absolute():
        base_path = (data_yaml.parent / base_path).resolve()

    data_cfg["path"] = str(base_path)

    with NamedTemporaryFile("w", suffix=".yaml", prefix="signpost_data_", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False, allow_unicode=True)
        return Path(f.name)


def sanitize_checkpoint_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: sanitize_checkpoint_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_checkpoint_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(sanitize_checkpoint_value(v) for v in value)
    return value


def resave_checkpoint_compatible(ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt = sanitize_checkpoint_value(ckpt)
    torch.save(
        ckpt,
        str(ckpt_path),
        pickle_protocol=4,
        _use_new_zipfile_serialization=False,
    )
    print(f"Re-saved checkpoint in compatibility format: {ckpt_path}")


def rename_best_checkpoint(save_dir: Path) -> None:
    weights_dir = save_dir / "weights"
    best_ckpt = weights_dir / "best.pt"
    yolo_best_ckpt = weights_dir / "yolo_best.pt"

    if not best_ckpt.exists():
        print(f"[WARN] best checkpoint not found: {best_ckpt}")
        return

    if yolo_best_ckpt.exists():
        yolo_best_ckpt.unlink()

    shutil.move(str(best_ckpt), str(yolo_best_ckpt))
    resave_checkpoint_compatible(yolo_best_ckpt)
    print(f"Renamed best checkpoint to: {yolo_best_ckpt}")


def main():
    args = parse_args()

    data_yaml = resolve_from_script(args.data)
    project_dir = resolve_from_script(args.project)

    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found: {data_yaml}")

    resolved_data_yaml = prepare_data_yaml(data_yaml)

    print(f"Using data: {data_yaml}")
    print(f"Resolved data config: {resolved_data_yaml}")
    print(f"Using model: {args.model}")
    print(f"Saving to: {project_dir / args.name}")

    model = YOLO(args.model)
    model.train(
        data=str(resolved_data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        pretrained=True,
        patience=args.patience,
        save_period=args.save_period,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
    )

    save_dir = Path(model.trainer.save_dir)
    rename_best_checkpoint(save_dir)


if __name__ == "__main__":
    main()
