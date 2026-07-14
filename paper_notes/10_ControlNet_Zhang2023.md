# 论文导读 10: ControlNet (Zhang et al., ICCV 2023)

**标题**：Adding Conditional Control to Text-to-Image Diffusion Models
**作者**：Lvmin Zhang, Anyi Rao, Maneesh Agrawala (Stanford)
**核心地位**：⭐⭐⭐⭐⭐ SD 生态最具影响力的"插件"工作

---

## 一、为什么必读

- SD 用户最常用的扩展（每个 prompt 都可能配 ControlNet）
- "Plug-in to frozen base model" 设计哲学影响深远
- 工程简洁、可复现、效果显著

> 不读 ControlNet 就不算理解 SD 工业生态。

---

## 二、阅读路线图

| 章节 | 优先级 |
|------|--------|
| §1 Intro | ✅ |
| **§3 Method** | 🔴 核心 |
| §3.1 Architecture | 🔴 核心 |
| §3.2 Zero conv | 🔴 核心 |
| §3.3 Training | ✅ |
| §4 Experiments | ✅ |

---

## 三、核心架构

```
SD UNet (locked):                ControlNet (trainable copy of encoder):
  z_t → enc1 → ... → dec → out    z_t + control → enc1' → ... → mid'
                                                      ↓ zero_conv
                                                   sum into SD UNet at each scale
```

**三个关键点**：
1. ControlNet 复制 SD UNet 的 encoder（约一半参数）
2. ControlNet 输入：`z_t` 加上 condition image（经 conv 提取）
3. ControlNet 的输出经 **zero conv** 后加到 SD UNet 对应层

---

## 四、Zero Convolution

```python
class ZeroConv(nn.Conv2d):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, kernel_size=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
```

**初始**：所有输出 = 0，ControlNet 不影响 SD。
**训练后**：weight 不再是 0，开始注入 condition 信息。

类似 LoRA 的 B=0 init 与 AdaLN-Zero 的零调制。

---

## 五、训练细节

- 训练参数：~500M（ControlNet 部分）
- 训练步数：~50K
- 数据：(condition image, target image, caption) 三元组
- 学习率：1e-5

数据集示例：
- Canny ControlNet：用 OpenCV Canny 对 LAION 图算边缘
- Depth ControlNet：用 MiDaS 算深度图
- Pose ControlNet：用 OpenPose 算骨架

---

## 六、常被误读

### 1. "ControlNet 改变了 SD 的权重？"

**没有**。ControlNet 是**冻结 SD + 添加副本**，SD 权重不变。这是它能"plug and play"的根本——同一 ControlNet 可用于任何 SD 1.5 fine-tune。

---

### 2. "ControlNet 的副本是新训的？"

**初始化**用 SD UNet 的 encoder 权重，但训练时**只更新副本**。这一点很关键：
- 副本与 SD 共享相同的特征空间
- 不会"打架"
- 训练比从头训 encoder 快几倍

---

### 3. "Multi-ControlNet 怎么工作？"

简单加性：
```python
output = sd_unet(z_t, t, prompt) \
       + 0.7 * canny_controlnet(z_t, t, prompt, canny_img) \
       + 0.5 * depth_controlnet(z_t, t, prompt, depth_img)
```

每个 ControlNet 独立训练，推理时叠加。权重需调参（典型 0.3-1.0）。

---

## 七、与其他 PEFT 对比

| 方法 | 参数量 | 训练数据 | 用途 |
|------|--------|---------|------|
| ControlNet | 500M | 几十万对 | 空间结构 |
| LoRA | ~10 MB | 几十张 | 风格/物体 |
| IP-Adapter | ~30 MB | 几百万对 | 图像 prompt |
| T2I-Adapter | ~80 MB | 几十万对 | 类似 ControlNet 但轻 |

ControlNet 最强但最贵；T2I-Adapter 是其简化版。

---

## 八、思考题

1. 为什么 ControlNet 只复制 encoder 而不复制整个 UNet？
2. 如果不用 zero conv，用标准 Kaiming init 训练，会出什么问题？
3. ControlNet 训练失败的常见原因有哪些？
4. 如何用 ControlNet 思路设计"3D shape → image"的扩展？

---

## 九、引用

```
@inproceedings{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  booktitle={ICCV},
  year={2023}
}
```
