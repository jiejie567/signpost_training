# eval

干净版评估导出脚本，只负责：

- 读取图片
- 运行 `YOLO`
- 可选运行 `CNN` 细分类
- 导出带框结果图
- 保存简单统计

不做这些事情：

- 不读取 GT
- 不计算 precision / recall / mAP
- 不导出历史样例目录

## Usage

默认会读取：

- `../yolo/dataset/images/val`
- `../yolo/runs/train/weights/best.pt`
- `../cnn/outputs/signpost_cnn_best.pth`

输出到：

- `eval/outputs/images`
- `eval/outputs/summary.json`
- `eval/outputs/per_image.json`

运行：

```bash
cd eval
python3 export_detection_images.py
```

只导出 YOLO 检测框：

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
