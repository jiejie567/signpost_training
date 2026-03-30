import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
CNN_DIR = (SCRIPT_DIR.parent / "cnn").resolve()
if str(CNN_DIR) not in sys.path:
    sys.path.insert(0, str(CNN_DIR))

from train_signpost_cnn import CLASS_NAMES, IMAGE_SIZE, SignpostCNN  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export YOLO or YOLO+CNN detection visualizations"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str((SCRIPT_DIR.parent / "yolo" / "dataset" / "images" / "val").resolve()),
        help="Image directory to visualize",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=str((SCRIPT_DIR.parent / "yolo" / "runs" / "train" / "weights" / "yolo_best.pt").resolve()),
        help="YOLO weights path",
    )
    parser.add_argument(
        "--cnn-weights",
        type=str,
        default=str((SCRIPT_DIR.parent / "cnn" / "outputs" / "signpost_cnn_best.pth").resolve()),
        help="Optional CNN weights path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str((SCRIPT_DIR / "outputs").resolve()),
        help="Directory for annotated images and summary json",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/mps")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.7)
    parser.add_argument("--cls-threshold", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Do not save images without any detections",
    )
    return parser.parse_args()


def build_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_gray(crop_bgr):
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(crop_gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor


def classify_crop(crop_bgr, model, device, threshold, use_half):
    tensor = preprocess_gray(crop_bgr).to(device)
    if use_half:
        tensor = tensor.half()

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    raw_name = CLASS_NAMES[int(pred.item())]
    confidence = float(conf.item())
    if confidence < threshold:
        return "unknown", confidence, raw_name
    return raw_name, confidence, raw_name


def draw_label(image, text, origin, color, font_scale=0.7, thickness=1):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    top = max(0, y - text_h - baseline - 6)
    bottom = max(text_h + baseline + 6, y)
    right = min(image.shape[1], x + text_w + 8)
    cv2.rectangle(image, (x, top), (right, bottom), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (x + 4, bottom - baseline - 3),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def find_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    yolo_weights = Path(args.yolo_weights)
    cnn_weights = Path(args.cnn_weights)
    output_dir = Path(args.output_dir)
    images_out_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")

    image_paths = find_images(images_dir)
    if args.limit > 0:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise RuntimeError(f"no images found in: {images_dir}")

    device = build_device(args.device)
    use_half = device.type == "cuda"
    yolo_model = YOLO(str(yolo_weights))

    cnn_model = None
    cnn_enabled = cnn_weights.exists()
    if cnn_enabled:
        cnn_model = SignpostCNN(num_classes=len(CLASS_NAMES)).to(device)
        state_dict = torch.load(str(cnn_weights), map_location=device)
        cnn_model.load_state_dict(state_dict)
        if use_half:
            cnn_model.half()
        cnn_model.eval()

    total_detections = 0
    saved_images = 0
    class_counts = Counter()
    raw_class_counts = Counter()
    per_image_rows = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        results = yolo_model.predict(
            source=image,
            conf=args.yolo_conf,
            imgsz=args.imgsz,
            verbose=False,
            device=str(device),
            half=use_half,
        )
        boxes = results[0].boxes
        vis = image.copy()
        detections = []

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image.shape[1], int(x2))
                y2 = min(image.shape[0], int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                det_conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                label = f"det {det_conf:.2f}"
                color = (0, 255, 0)
                cls_name = None
                cls_conf = None

                if cnn_model is not None:
                    crop = image[y1:y2, x1:x2]
                    if crop.size != 0:
                        cls_name, cls_conf, raw_cls_name = classify_crop(
                            crop,
                            cnn_model,
                            device,
                            args.cls_threshold,
                            use_half,
                        )
                        raw_class_counts[raw_cls_name] += 1
                        if cls_name == "unknown":
                            color = (0, 165, 255)
                            label = f"unknown<{raw_cls_name}> | det {det_conf:.2f} | cls {cls_conf:.2f}"
                        else:
                            class_counts[cls_name] += 1
                            label = f"{cls_name} | det {det_conf:.2f} | cls {cls_conf:.2f}"

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "det_conf": det_conf,
                        "cls_name": cls_name,
                        "cls_conf": cls_conf,
                    }
                )

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                draw_label(vis, label, (x1, max(24, y1)), color)

        if detections or not args.skip_empty:
            out_path = images_out_dir / image_path.name
            cv2.imwrite(str(out_path), vis)
            saved_images += 1

        total_detections += len(detections)
        per_image_rows.append(
            {
                "image": image_path.name,
                "num_detections": len(detections),
                "detections": detections,
            }
        )

    summary = {
        "images_dir": str(images_dir),
        "yolo_weights": str(yolo_weights),
        "cnn_weights": str(cnn_weights) if cnn_enabled else None,
        "cnn_enabled": cnn_enabled,
        "device": str(device),
        "num_images": len(image_paths),
        "saved_images": saved_images,
        "total_detections": total_detections,
        "classified_counts": dict(class_counts),
        "raw_top1_counts": dict(raw_class_counts),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "per_image.json").write_text(
        json.dumps(per_image_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[INFO] Images processed: {len(image_paths)}")
    print(f"[INFO] Images saved: {saved_images}")
    print(f"[INFO] Total detections: {total_detections}")
    print(f"[INFO] Output dir: {output_dir}")
    if cnn_enabled:
        print(f"[INFO] Classified counts: {dict(class_counts)}")
    else:
        print("[INFO] CNN disabled: cnn weights not found, exported YOLO-only visualizations")


if __name__ == "__main__":
    main()
