# signpost_training

独立版 signpost 训练工程，只保留 `YOLO` 检测和 `CNN` 分类训练、阈值分析与可视化评估所需的脚本和数据。

## Quick Start

安装依赖：

```bash
pip install -r requirements.txt
```

## Structure

- `yolo/train_yolo.py`
- `yolo/dataset/`
- `cnn/train_signpost_cnn.py`
- `cnn/analyze_threshold.py`
- `cnn/dataset/`
- `eval/export_detection_images.py`

没有保留这些无关内容：

- 实时推理脚本
- ROS 节点
- 评估导出图
- 历史训练产物
- `.DS_Store` / `__pycache__` / `*.cache`
- 旧的 MobileNet 训练脚本

## YOLO

单类检测，只检测路牌位置（不区分类型）。

目录：

- `yolo/dataset/images/train`
- `yolo/dataset/images/val`
- `yolo/dataset/labels/train`
- `yolo/dataset/labels/val`
- `yolo/dataset/data.yaml`

训练：

```bash
cd yolo
python3 train_yolo.py --device cpu
```

默认输出到：

```text
yolo/runs/train
```

## CNN

9 类分类器，输入为 YOLO 裁出的路牌区域（128×128 灰度 + Otsu 二值化）。

| index | 类别 | 说明 |
|-------|------|------|
| 0 | S1 | |
| 1 | S2 | |
| 2 | S3 | |
| 3 | S5 | |
| 4 | S6 | |
| 5 | S8 | |
| 6 | S9 | |
| 7 | S10 | |
| 8 | others | 训练集以外的未知路牌类型 |

推理时预测为 `others` 或 softmax 置信度低于阈值（默认 0.7）时，输出 `unknown`。

目录：

- `cnn/dataset/train/S1` ... `S10`, `others`
- `cnn/dataset/val/S1` ... `S10`, `others`

训练：

```bash
cd cnn
python3 train_signpost_cnn.py --split-dir dataset
```

阈值分析：

```bash
python3 analyze_threshold.py
```

默认输出到：

```text
cnn/outputs/signpost_cnn_best.pth
```

## Eval

导出检测结果图（YOLO 检测框 + CNN 分类标注）：

```bash
cd eval
python3 export_detection_images.py
```

输出目录：

```text
eval/outputs/images/
eval/outputs/summary.json
eval/outputs/per_image.json
```
