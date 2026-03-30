"""
分析模型在验证集上的置信度分布，帮助选择 unknown 拒识阈值。

用法:
    cd cnn
    python analyze_threshold.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# 从训练脚本导入模型和常量
from train_signpost_cnn import (
    SignpostCNN, CLASS_NAMES, CLASS_TO_IDX, IMAGE_SIZE, get_val_transform
)

def main():
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "outputs" / "signpost_cnn_best.pth"
    val_dir = script_dir / "split" / "val"

    if not model_path.exists():
        print(f"[ERROR] 模型不存在: {model_path}")
        return

    # 加载模型
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model = SignpostCNN(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    transform = get_val_transform()

    # 收集每个样本的置信度和预测结果
    results = []  # (true_class, pred_class, confidence, correct)

    for cls_name in CLASS_NAMES:
        cls_dir = val_dir / cls_name
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            img = Image.open(img_path).convert("L")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=1)
                conf, pred_idx = probs.max(dim=1)

            conf = conf.item()
            pred_name = CLASS_NAMES[pred_idx.item()]
            correct = (pred_name == cls_name)
            results.append((cls_name, pred_name, conf, correct))

    if not results:
        print("[ERROR] 验证集为空")
        return

    # ── 总体统计 ──
    confs = [r[2] for r in results]
    correct_confs = [r[2] for r in results if r[3]]
    wrong_confs = [r[2] for r in results if not r[3]]

    print("=" * 60)
    print(" 置信度分布分析")
    print("=" * 60)
    print(f"\n总样本数: {len(results)}")
    print(f"正确预测: {len(correct_confs)}")
    print(f"错误预测: {len(wrong_confs)}")
    print(f"原始准确率: {len(correct_confs) / len(results):.4f}")

    print(f"\n{'':>10} {'总体':>10} {'正确':>10} {'错误':>10}")
    print("-" * 45)
    print(f"{'平均':>10} {np.mean(confs):>10.4f} "
          f"{np.mean(correct_confs):>10.4f} "
          f"{np.mean(wrong_confs) if wrong_confs else 'N/A':>10}")
    print(f"{'最小':>10} {np.min(confs):>10.4f} "
          f"{np.min(correct_confs):>10.4f} "
          f"{np.min(wrong_confs) if wrong_confs else 'N/A':>10}")
    print(f"{'中位数':>10} {np.median(confs):>10.4f} "
          f"{np.median(correct_confs):>10.4f} "
          f"{np.median(wrong_confs) if wrong_confs else 'N/A':>10}")

    # ── 不同阈值下的效果 ──
    print(f"\n{'阈值':>8} {'准确率':>10} {'拒识率':>10} {'误分类率':>10} {'说明':>20}")
    print("-" * 65)

    for threshold in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
        accepted = [(r[2], r[3]) for r in results if r[2] >= threshold]
        rejected = [r for r in results if r[2] < threshold]

        if accepted:
            acc = sum(1 for _, c in accepted if c) / len(accepted)
        else:
            acc = 0.0
        reject_rate = len(rejected) / len(results)
        # 误分类 = 被接受但预测错误
        misclass = sum(1 for _, c in accepted if not c)
        misclass_rate = misclass / len(results)

        note = ""
        if reject_rate < 0.05 and acc > 0.95:
            note = "<-- 推荐"
        elif reject_rate < 0.10 and acc > 0.98:
            note = "<-- 高精度推荐"

        print(f"{threshold:>8.2f} {acc:>10.4f} {reject_rate:>10.4f} {misclass_rate:>10.4f} {note:>20}")

    # ── 每个类别的置信度 ──
    print(f"\n{'类别':>6} {'样本数':>6} {'准确率':>8} {'平均置信度':>10} {'最低置信度':>10}")
    print("-" * 50)
    class_results = defaultdict(list)
    for r in results:
        class_results[r[0]].append(r)

    for cls_name in CLASS_NAMES:
        if cls_name not in class_results:
            continue
        cr = class_results[cls_name]
        n = len(cr)
        acc = sum(1 for r in cr if r[3]) / n
        avg_conf = np.mean([r[2] for r in cr])
        min_conf = np.min([r[2] for r in cr])
        print(f"{cls_name:>6} {n:>6} {acc:>8.4f} {avg_conf:>10.4f} {min_conf:>10.4f}")

    # ── 列出错误样本 ──
    if wrong_confs:
        print(f"\n错误预测详情:")
        print(f"{'真实':>6} {'预测':>6} {'置信度':>10}")
        print("-" * 28)
        for r in sorted(results, key=lambda x: x[2]):
            if not r[3]:
                print(f"{r[0]:>6} {r[1]:>6} {r[2]:>10.4f}")


if __name__ == "__main__":
    main()
