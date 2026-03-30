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

目录：

- `cnn/dataset/S1`
- `cnn/dataset/S2`
- `cnn/dataset/S3`
- `cnn/dataset/S5`
- `cnn/dataset/S6`
- `cnn/dataset/S8`
- `cnn/dataset/S9`
- `cnn/dataset/S10`

训练：

```bash
cd cnn
python3 train_signpost_cnn.py
```

阈值分析：

```bash
python3 analyze_threshold.py
```

默认输出到：

```text
cnn/outputs
```

## Eval

导出检测结果图：

```bash
cd eval
python3 export_detection_images.py
```

输出目录：

```text
eval/outputs
```

## Publish Notes

当前仓库已经去掉了 ROS 代码、实时推理脚本和历史训练产物，适合单独初始化为一个公开 GitHub 仓库。
