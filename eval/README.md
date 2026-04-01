# eval

导出检测+分类可视化图，输出 JSON 统计。

## 默认路径

| 参数 | 默认值 |
|------|--------|
| `--images-dir` | `../yolo/dataset/images/val` |
| `--yolo-weights` | `../yolo/runs/train/weights/yolo_best.pt` |
| `--cnn-weights` | `../cnn/outputs/signpost_cnn_best.pth` |
| `--output-dir` | `eval/outputs` |

## 运行

```bash
cd eval
python3 export_detection_images.py
```

只导出 YOLO 检测框（不做 CNN 分类）：

```bash
python3 export_detection_images.py --cnn-weights /path/not_exists.pth
```

只处理前 20 张图：

```bash
python3 export_detection_images.py --limit 20
```

跳过无检测图片：

```bash
python3 export_detection_images.py --skip-empty
```

## 输出

- `eval/outputs/images/` — 带框结果图
- `eval/outputs/summary.json` — 总体统计
- `eval/outputs/per_image.json` — 每张图的检测详情

## 分类逻辑

CNN 预测结果为 `others` 或置信度低于 `--cls-threshold`（默认 0.7）时，标注为 `unknown`（橙色框），否则显示具体类别（绿色框）。
