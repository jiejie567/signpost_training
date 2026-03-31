# Prototype-based OOD Detection

CNN 分类器在遇到训练集以外的路牌类别时，会用特征距离判断是否为未知类（unknown），而非依赖 softmax 置信度。

## 原理

模型 `SignpostCNN` 的 `features` 层将输入图像映射为 2048 维特征向量。同一类别的图像在该空间中聚集，不同类别的图像分散。

训练后，对每个已知类别计算验证集特征的均值，得到 8 个 **class prototype**（类中心）。推理时：

1. 提取输入图像的 2048 维特征
2. 计算到 8 个 prototype 的欧氏距离
3. 取最小距离 `min_dist`
4. 若 `min_dist > ood_threshold`，判为 `unknown`；否则取距离最近的类

`ood_threshold` 为验证集所有样本 `min_dist` 的第 99 百分位数（当前值约 27.28）。

## 文件

| 文件 | 说明 |
|------|------|
| `cnn/outputs/signpost_cnn_best.pth` | CNN 模型权重 |
| `cnn/outputs/prototypes.npz` | prototype 中心 + ood_threshold |

`prototypes.npz` 内容：

```python
import numpy as np
data = np.load("prototypes.npz", allow_pickle=True)

data["prototypes"]   # shape (8, 2048), float32，每行是一个类的特征中心
data["class_names"]  # ['S1', 'S2', 'S3', 'S5', 'S6', 'S8', 'S9', 'S10']
data["threshold"]    # float32，OOD 判断阈值
```

## 推理接口

### 模型定义（`cnn/train_signpost_cnn.py`）

```python
class SignpostCNN(nn.Module):
    def extract_features(self, x) -> torch.Tensor:
        """
        输入: x, shape (B, 1, 128, 128), float32, 值域 [0, 1]
        输出: shape (B, 2048), float32
        """

    def forward(self, x) -> torch.Tensor:
        """
        输出: shape (B, 8), logits（不经过 softmax）
        """
```

### 完整推理流程

```python
import numpy as np
import torch
import torch.nn.functional as F

# 1. 加载
model = SignpostCNN(num_classes=8)
model.load_state_dict(torch.load("cnn/outputs/signpost_cnn_best.pth",
                                  map_location="cpu", weights_only=False))
model.eval()

data = np.load("cnn/outputs/prototypes.npz", allow_pickle=True)
prototypes = data["prototypes"].astype(np.float32)  # (8, 2048)
class_names = list(data["class_names"])
ood_threshold = float(data["threshold"])

# 2. 预处理（灰度图，128×128，值域 [0,1]）
# img_gray: np.ndarray, shape (128, 128), dtype float32
tensor = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128)

# 3. 推理
with torch.no_grad():
    feat = model.extract_features(tensor).numpy()[0]  # (2048,)

dists = np.linalg.norm(prototypes - feat, axis=1)  # (8,)
pred_idx = int(dists.argmin())
min_dist = float(dists[pred_idx])

if min_dist > ood_threshold:
    result = "unknown"
else:
    result = class_names[pred_idx]
```

### 类别映射

```python
CLASS_NAMES = ["S1", "S2", "S3", "S5", "S6", "S8", "S9", "S10"]
# index:         0     1     2     3     4     5     6     7
```

注意：没有 S4 和 S7，index 和类名不连续。

## 重新生成 prototypes

每次重新训练 CNN 后需要重新生成：

```bash
cd cnn
python build_prototypes.py
```

输出自动覆盖 `cnn/outputs/prototypes.npz`。

## 无 prototypes 时的降级行为

`eval/export_detection_images.py` 在 `prototypes.npz` 不存在时，自动回退到 softmax 置信度阈值（`--cls-threshold`，默认 0.7），不报错。
