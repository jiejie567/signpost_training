"""
计算并保存每个类别的 prototype（特征中心）和 OOD 判断阈值。

用法:
    cd cnn
    python build_prototypes.py

输出: cnn/outputs/prototypes.npz
  - prototypes: (num_classes, 2048) 每个类的特征均值
  - class_names: CLASS_NAMES 列表
  - threshold:   验证集 min_dist 的第 99 百分位数
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_signpost_cnn import (
    CLASS_NAMES, SignpostCNN, SignpostDataset, get_val_transform, build_device,
)


def main():
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "outputs" / "signpost_cnn_best.pth"
    val_dir = script_dir / "dataset" / "val"
    output_path = script_dir / "outputs" / "prototypes.npz"

    device = build_device()
    print(f"[INFO] Device: {device}")

    model = SignpostCNN(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=False))
    model.eval()

    dataset = SignpostDataset(str(val_dir), transform=get_val_transform())
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"[INFO] Val samples: {len(dataset)}")

    # 提取所有验证集特征
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            feats = model.extract_features(images.to(device))
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    all_feats = torch.cat(all_feats, dim=0).numpy()   # (N, 2048)
    all_labels = torch.cat(all_labels, dim=0).numpy() # (N,)

    # 计算每个类的 prototype（特征均值）
    num_classes = len(CLASS_NAMES)
    prototypes = np.zeros((num_classes, all_feats.shape[1]), dtype=np.float32)
    for cls_idx in range(num_classes):
        mask = all_labels == cls_idx
        if mask.sum() == 0:
            print(f"[WARN] No samples for class {CLASS_NAMES[cls_idx]}")
            continue
        prototypes[cls_idx] = all_feats[mask].mean(axis=0)
        print(f"  {CLASS_NAMES[cls_idx]:>4s}: {mask.sum()} samples, "
              f"mean_norm={np.linalg.norm(prototypes[cls_idx]):.2f}")

    # 计算每个验证集样本到最近 prototype 的距离
    min_dists = []
    for i, (feat, label) in enumerate(zip(all_feats, all_labels)):
        dists = np.linalg.norm(prototypes - feat, axis=1)  # (num_classes,)
        min_dists.append(dists.min())

    min_dists = np.array(min_dists)
    threshold = float(np.percentile(min_dists, 99))
    print(f"\n[INFO] min_dist stats: mean={min_dists.mean():.4f}, "
          f"p95={np.percentile(min_dists, 95):.4f}, "
          f"p99={min_dists.max():.4f}")
    print(f"[INFO] OOD threshold (p99): {threshold:.4f}")

    np.savez(
        str(output_path),
        prototypes=prototypes,
        class_names=np.array(CLASS_NAMES),
        threshold=np.float32(threshold),
    )
    print(f"[INFO] Saved to: {output_path}")


if __name__ == "__main__":
    main()
