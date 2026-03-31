"""
Signpost CNN Classifier — 轻量自定义 CNN
========================================
8 类 sign (S1, S2, S3, S5, S6, S8, S9, S10) + 置信度拒识 unknown

用法:
    python train_signpost_cnn.py                        # 训练
    python train_signpost_cnn.py --eval-only            # 仅评估已有模型
    python train_signpost_cnn.py --predict img.jpg      # 单张推理
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ──────────────────────────────────────────────
# 1. 模型定义
# ──────────────────────────────────────────────

class SignpostCNN(nn.Module):
    """
    轻量 4 层 CNN，适合小数据集的黑白几何图案分类。

    输入:  1×64×64  (灰度)
    输出:  num_classes 维 logits

    结构:
        Conv-BN-ReLU-Pool  ×4  →  AdaptiveAvgPool  →  FC-ReLU-Dropout  →  FC

    参数量约 ~800K，相比 MobileNetV3-Small (2.5M) 轻很多。
    """

    def __init__(self, num_classes: int = 8):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1×64×64 → 32×32×32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32×32×32 → 64×16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64×16×16 → 128×8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128×8×8 → 128×4×4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # 固定输出 4×4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def extract_features(self, x):
        """返回 features 层输出的展平向量 (B, 128*4*4=2048)，用于 prototype 距离计算。"""
        return self.features(x).flatten(1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────
# 2. 数据集
# ──────────────────────────────────────────────

# 8 个已知类别 (排序确保 index 稳定)
CLASS_NAMES = ["S1", "S2", "S3", "S5", "S6", "S8", "S9", "S10"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 128  # 输入尺寸


class SignpostDataset(Dataset):
    """
    从 ImageFolder 格式的目录加载数据。
    只加载 CLASS_NAMES 中定义的类别，忽略未定义文件夹。
    自动转灰度 + resize。
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # list of (path, label_idx)

        for cls_name in CLASS_NAMES:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.is_dir():
                continue
            for fname in sorted(cls_dir.iterdir()):
                if fname.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((str(fname), CLASS_TO_IDX[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # 转灰度
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)

        return img, label


# ──────────────────────────────────────────────
# 3. 数据增强 (torchvision transforms)
# ──────────────────────────────────────────────

from torchvision import transforms


def get_train_transform():
    """
    训练时数据增强。
    注意: 不使用水平翻转，因为 S3(⊢) 和 S8(⊢菱形) 等符号翻转后会和其他类混淆。
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),  # [0, 1], shape: 1×H×W for grayscale
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


# ──────────────────────────────────────────────
# 4. Train / Val 划分
# ──────────────────────────────────────────────

def split_train_val(src_dir: str, dst_train_dir: str, dst_val_dir: str,
                    val_ratio: float = 0.2, seed: int = 42):
    """
    从 src_dir (ImageFolder) 按类别分层划分 train/val 到 dst 目录。
    使用 symlink 避免复制文件。如果 dst 已有数据则跳过。
    """
    src = Path(src_dir)
    dst_train = Path(dst_train_dir)
    dst_val = Path(dst_val_dir)

    # 检查是否已经划分过
    already_done = False
    for cls_name in CLASS_NAMES:
        if (dst_val / cls_name).is_dir() and any((dst_val / cls_name).iterdir()):
            already_done = True
            break
    if already_done:
        print("[INFO] Train/Val split already exists, skipping.")
        return

    rng = random.Random(seed)

    for cls_name in CLASS_NAMES:
        cls_src = src / cls_name
        if not cls_src.is_dir():
            continue

        files = sorted([f for f in cls_src.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_ratio))
        val_files = set(f.name for f in files[:n_val])

        for split_dir in [dst_train / cls_name, dst_val / cls_name]:
            split_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            if f.name in val_files:
                dst = dst_val / cls_name / f.name
            else:
                dst = dst_train / cls_name / f.name
            if not dst.exists():
                # 使用 symlink 避免数据复制
                try:
                    dst.symlink_to(f.resolve())
                except OSError:
                    shutil.copy2(str(f), str(dst))

    # 打印统计
    print("[INFO] Split statistics:")
    for cls_name in CLASS_NAMES:
        n_train = len(list((dst_train / cls_name).iterdir())) if (dst_train / cls_name).exists() else 0
        n_val = len(list((dst_val / cls_name).iterdir())) if (dst_val / cls_name).exists() else 0
        print(f"  {cls_name:>4s}: train={n_train:>3d}, val={n_val:>3d}")


# ──────────────────────────────────────────────
# 5. 训练 & 评估
# ──────────────────────────────────────────────

def build_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_state_dict_compatible(state_dict, path: Path):
    """
    Save with legacy torch serialization for better compatibility with older
    Python/PyTorch environments.
    """
    torch.save(
        state_dict,
        str(path),
        pickle_protocol=4,
        _use_new_zipfile_serialization=False,
    )


def make_weighted_sampler(dataset):
    """类别均衡采样器，应对样本不均衡 (S2=54 vs S5=106)。"""
    label_counts = {}
    for _, label in dataset.samples:
        label_counts[label] = label_counts.get(label, 0) + 1

    weights = []
    for _, label in dataset.samples:
        weights.append(1.0 / label_counts[label])

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def compute_confusion_matrix(model, loader, device, num_classes):
    model.eval()
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        for t, p in zip(labels.cpu(), preds.cpu()):
            matrix[t, p] += 1
    return matrix


def print_confusion_matrix(matrix, class_names):
    w = max(len(n) for n in class_names)
    w = max(w, 6)
    header = f"{'':>{w}} " + " ".join(f"{n:>{w}}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = " ".join(f"{int(matrix[i, j]):>{w}d}" for j in range(len(class_names)))
        print(f"{name:>{w}} {row}")


def save_results(model, loader, device, output_dir, class_names):
    """保存混淆矩阵和类别映射。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 类别映射
    mapping = {str(i): name for i, name in enumerate(class_names)}
    (output_dir / "class_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 混淆矩阵
    cm = compute_confusion_matrix(model, loader, device, len(class_names))
    print("\n[Confusion Matrix]")
    print_confusion_matrix(cm, class_names)

    # CSV
    lines = [",".join(["true\\pred"] + class_names)]
    for i, name in enumerate(class_names):
        lines.append(",".join([name] + [str(int(cm[i, j])) for j in range(len(class_names))]))
    (output_dir / "confusion_matrix.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 尝试画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm.numpy(), cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(str(output_dir / "confusion_matrix.png"), dpi=150)
        plt.close(fig)
        print(f"[INFO] Confusion matrix plot saved to {output_dir / 'confusion_matrix.png'}")
    except Exception as e:
        print(f"[WARN] Could not plot confusion matrix: {e}")


# ──────────────────────────────────────────────
# 6. 推理 (置信度拒识)
# ──────────────────────────────────────────────

def predict_single(model, image_path: str, device, threshold: float = 0.7):
    """
    对单张图片推理。
    返回 (predicted_class_name, confidence)。
    如果 confidence < threshold，返回 ("unknown", confidence)。
    """
    model.eval()
    img = Image.open(image_path).convert("L")
    transform = get_val_transform()
    img_tensor = transform(img).unsqueeze(0).to(device)  # 1×1×64×64

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)
        confidence = confidence.item()
        pred_idx = pred_idx.item()

    if confidence < threshold:
        return "unknown", confidence
    return CLASS_NAMES[pred_idx], confidence


# ──────────────────────────────────────────────
# 7. 主函数
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train signpost CNN classifier")
    parser.add_argument("--data-root", type=str,
                        default="dataset",
                        help="原始 ImageFolder 数据目录 (含 S1/, S2/, ...)")
    parser.add_argument("--split-dir", type=str,
                        default="split",
                        help="划分后的 train/val 存放目录")
    parser.add_argument("--output-dir", type=str,
                        default="outputs",
                        help="训练输出目录 (模型、混淆矩阵等)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (0=不启用)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="推理时 unknown 拒识阈值")
    parser.add_argument("--eval-only", action="store_true",
                        help="跳过训练，仅评估已有模型")
    parser.add_argument("--predict", type=str, default=None,
                        help="单张图片推理路径")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 路径处理 — 支持相对路径 (相对于脚本所在目录)
    script_dir = Path(__file__).resolve().parent
    data_root = Path(args.data_root) if Path(args.data_root).is_absolute() else script_dir / args.data_root
    split_dir = Path(args.split_dir) if Path(args.split_dir).is_absolute() else script_dir / args.split_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else script_dir / args.output_dir
    model_path = output_dir / "signpost_cnn_best.pth"

    device = build_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Classes: {CLASS_NAMES} ({len(CLASS_NAMES)} classes)")
    print(f"[INFO] Input size: {IMAGE_SIZE}×{IMAGE_SIZE} grayscale")

    # ── 单张推理 ──
    if args.predict:
        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path}")
            return
        model = SignpostCNN(num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=False))
        cls_name, conf = predict_single(model, args.predict, device, args.threshold)
        print(f"Prediction: {cls_name}  (confidence={conf:.4f}, threshold={args.threshold})")
        return

    # ── 数据划分 ──
    split_train = split_dir / "train"
    split_val = split_dir / "val"
    split_train_val(str(data_root), str(split_train), str(split_val),
                    val_ratio=args.val_ratio, seed=args.seed)

    # ── 加载数据 ──
    train_dataset = SignpostDataset(str(split_train), transform=get_train_transform())
    val_dataset = SignpostDataset(str(split_val), transform=get_val_transform())
    print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    sampler = make_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # ── 模型 ──
    model = SignpostCNN(num_classes=len(CLASS_NAMES)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params:,}")

    # ── 仅评估 ──
    if args.eval_only:
        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path}")
            return
        model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=False))
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"[EVAL] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        save_results(model, val_loader, device, str(output_dir), CLASS_NAMES)
        return

    # ── 训练 ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    patience_counter = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'TrainAcc':>10} {'ValLoss':>10} {'ValAcc':>10} {'LR':>10}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6d} {train_loss:>10.4f} {train_acc:>10.4f} "
              f"{val_loss:>10.4f} {val_acc:>10.4f} {lr_now:>10.6f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_state_dict_compatible(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n[INFO] Early stopping at epoch {epoch} (patience={args.patience})")
            break

        scheduler.step()

    print(f"\n[INFO] Best Val Acc: {best_val_acc:.4f}")
    print(f"[INFO] Model saved to: {model_path}")

    # ── 最终评估 ──
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=False))
    save_results(model, val_loader, device, str(output_dir), CLASS_NAMES)
    print("[DONE]")


if __name__ == "__main__":
    main()
