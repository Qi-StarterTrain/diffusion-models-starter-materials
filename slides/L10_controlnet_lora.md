# L10: ControlNet 与 LoRA 进阶（W9）

> **目标**：理解扩散模型的"插件式"控制——不动主干、加少量参数实现条件控制与个性化
> **前置**：L08 (CFG), L09 (LDM)
> **核心问题**：当我们已经有 SD 这种大模型，怎么"低成本"加新能力？

---

## §1 课程定位

P0-P1 部分：从零造一个扩散模型。
P2 开始：在已有大模型上**做加法**。

这是工业界真实的工作流：
- 训一次 SD 花上百万美金 → 不能动
- 但用户需求多变（特定画风、特定姿态、特定 IP）
- 解决方案：**Plug-in 思路** → ControlNet、LoRA、Textual Inversion、IP-Adapter…

**核心三问**：
1. 加什么模块？（架构）
2. 训什么参数？（trainable subset）
3. 推理时怎么注入？（forward pass 修改）

---

## §2 ControlNet（Zhang et al., 2023）

### 2.1 动机

CFG 只能"文本引导"。如果我想：
- 生成一只猫，但**姿态**像我画的草图
- 生成一座建筑，但**深度结构**像参考图
- 生成一个人，但**关键点**像我提供的骨架

→ 需要**空间结构条件**。

---

### 2.2 架构核心

ControlNet 的关键 trick 三个：

**1. Trainable copy of UNet encoder**
- 复制原 SD UNet 的 encoder（约一半参数）
- 副本接受 condition image 作为额外输入

**2. Zero Convolution**
- 副本输出通过 1×1 卷积加到原 UNet 对应位置
- 1×1 卷积**初始化为 0**

**3. Locked main UNet**
- 原 SD UNet 权重冻结，只训副本 + zero conv

```
原 SD UNet:           ControlNet 副本:
  ↓ enc1               ↓ enc1' (copy)
  ↓ enc2     ←——      ↓ enc2' (copy)
  ↓ enc3     ←——      ↓ enc3' (copy)
  ↓ mid      ←——      ↓ mid'  (copy)
  ↓ dec3
  ↓ dec2
  ↓ dec1
```

副本与原 UNet 通过 zero conv 在每个分辨率求和。

---

### 2.3 Zero Convolution 为什么有效

**核心问题**：把一个未训练的副本"接"到训好的 SD 上，会不会破坏 SD？

**答案**：用 zero conv 让初始连接强度为 0。

```python
# Pseudo
class ZeroConv(nn.Conv2d):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, kernel_size=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
```

**初始**：副本输出乘 0 = 没有影响，等价于原 SD。
**训练几步后**：zero conv 权重不再是 0，开始注入 condition 信息。

> **类比 ResNet**：identity skip 让深层网络初始等价于浅层，再慢慢学。ControlNet 让"扩展模型"初始等价于原模型。

---

### 2.4 训练

数据集：(condition image, target image, caption) 三元组。

```python
loss = MSE(eps_pred, true_eps)
# eps_pred = unet_main(z_t, t, text_emb) + controlnet(z_t, t, text_emb, control_img)
```

只训练 controlnet 的参数（约 500M for SD 1.5 的 controlnet）。

---

### 2.5 推理（CFG + ControlNet）

```python
# Batched CFG with ControlNet
eps_cond = unet(z, t, text_cond) + control_strength * controlnet(z, t, text_cond, condition_img)
eps_uncond = unet(z, t, text_uncond) + control_strength * controlnet(z, t, text_uncond, condition_img)
eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
```

注意：
- `control_strength` 控制 controlnet 的影响强度（0 = 关闭，1 = 默认）
- 可以**多个 controlnet 叠加**（canny + depth + pose）

---

### 2.6 常见 condition 类型

| 类型 | 提取工具 | 用途 |
|------|---------|------|
| Canny edge | OpenCV cv2.Canny | 保持轮廓 |
| Depth | MiDaS / ZoeDepth | 保持 3D 结构 |
| Pose (OpenPose) | OpenPose detector | 保持人物姿态 |
| Segmentation | SAM / DeepLab | 保持区域划分 |
| Normal map | 法线估计网络 | 保持表面方向 |
| Scribble | 手画 | 用户草图 |

---

## §3 LoRA（Hu et al., 2021）

### 3.1 动机

ControlNet 加 500M 参数仍然不小。如果我只想让 SD 学会**画我家猫的风格**呢？

LoRA：用极小（几 MB）的 adapter 实现微调。

---

### 3.2 数学核心

对任意线性层 $W \in \mathbb{R}^{d \times k}$，LoRA 把更新参数化为低秩矩阵：

$$W_{\text{new}} = W + \Delta W = W + B A$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，**rank $r$ 远小于 $\min(d, k)$**。

参数量从 $d \times k$ 降到 $r(d + k)$。例如 $d = k = 1024, r = 8$：从 1M → 16K（**减少 60 倍**）。

---

### 3.3 训练初始化

$$A \sim \mathcal{N}(0, \sigma^2), \quad B = 0$$

所以 $\Delta W = BA = 0$，初始等价于原模型（与 zero conv 同思路）。

训练时：原 $W$ 冻结，只训 $A, B$。

---

### 3.4 推理时合并

$$W_{\text{final}} = W + B A \quad \text{(融合，无额外计算)}$$

或者保留为 adapter 形式（运行时可热切换）：
$$y = Wx + BAx = Wx + B(Ax)$$

第二种允许动态混合多个 LoRA（如 `LoRA_animeStyle * 0.8 + LoRA_pose * 0.3`）。

---

### 3.5 在 SD 中的应用

SD UNet 的哪些层加 LoRA？
- **Cross-attention 的 Q, K, V, output**（最常见，效果最好）
- 也可选 self-attention（次要）
- 不建议 FFN（参数多但增益小）

典型配置：
```python
target_modules = ['to_q', 'to_k', 'to_v', 'to_out.0']  # cross-attn
rank = 8 - 32
alpha = rank (scaling factor)
```

---

### 3.6 LoRA 训练规模

| 任务 | 数据量 | 训练步数 | 显存 |
|------|--------|---------|------|
| 单 object（dog/sks） | 5-30 张 | 500-2000 | 8 GB |
| 单画风（特定艺术家） | 20-100 张 | 2000-5000 | 8 GB |
| 多风格融合 | 数百张 | 10K+ | 12 GB |

—— 比 full fine-tune（数百小时、12+ GB）便宜两个数量级。

---

## §4 ControlNet vs LoRA：本质对比

| 维度 | ControlNet | LoRA |
|------|-----------|------|
| 目标 | 增加新**条件**通道 | 微调现有**权重** |
| 改变什么 | UNet forward（加副本输出） | UNet 权重（加 ΔW） |
| 训练参数 | ~500M | 几 MB |
| 数据量 | 几万对 | 几十张 |
| 用途 | 空间结构控制 | 风格/物体记忆 |
| 推理开销 | UNet 计算量 +50% | 几乎为 0 |

**组合使用**：实际工业流水线通常 `ControlNet + LoRA + CFG` 全开。

---

## §5 其他 PEFT 方法（概览）

### 5.1 Textual Inversion (Gal et al., 2022)

只学一个**新的 token embedding**（768 维向量）。
- 参数量：768
- 学不到结构能力，但能"记一个 concept"
- 用法：在 prompt 里用 `<concept>` 触发

### 5.2 DreamBooth (Ruiz et al., 2022)

Full fine-tune SD UNet，但加 **class-specific prior preservation loss** 防止 catastrophic forgetting。
- 质量好但代价大（10+ GB 显存）
- 适合"重要 IP"的高质量定制

### 5.3 IP-Adapter (Ye et al., 2023)

把图像作为"prompt"。用一个 image encoder + 小 adapter 注入到 cross-attention。
- 支持 "image variations"
- 与 ControlNet 互补（IP 决定 what，ControlNet 决定 where）

### 5.4 Custom Diffusion (Kumari et al., 2023)

只训 cross-attention 的 K, V 矩阵。
- 比 full fine-tune 快 6×
- 多 concept 融合优于 LoRA

---

## §6 工程实战要点

### 6.1 ControlNet 训练
- 数据集要 **平衡 condition 和 caption**：condition 不能完全决定 target（否则模型只学复印），也不能完全不相关
- batch size 至少 128
- 训练 ~50K steps（SD 1.5 controlnet）

### 6.2 LoRA 训练
- rank 大未必好。简单任务 r=4 足够，复杂任务 r=16-32
- learning rate：1e-4 起步（不要用 SD 训练的 1e-5）
- 训练步数：少量但密集（500-2000 步）
- 提早停训练，**catastrophic forgetting 比 underfit 更可怕**

### 6.3 LoRA 调试
- Loss 不降：检查是否真的把 LoRA 加到了 cross-attn（看 trainable params）
- 生成"过拟合"训练集：rank 太大或训练过久
- 影响其他 prompt：用 trigger word（"sks" 等）让 LoRA 只在特定 prompt 下激活

---

## §7 与 VLA / 具身的连接

这套"主干 + 插件"的思路在**具身智能**领域同样适用：

- **基础视觉-语言模型**（如 CLIP, SigLIP）+ ControlNet-like adapter 接入 action head
- **Pretrained 大 policy**（如 Octo, OpenVLA）+ LoRA 微调到特定 robot
- **RT-2 / Pi-0**：Vision-Language model + action token adapter

ControlNet 与 LoRA 的设计哲学（"加少不动多，初始等价"）在 robotics 里直接复用。

---

## §8 课后任务

### 必做
1. 跑通本周 Project 3 的 `04_controlnet_demo.ipynb`
2. 用 Canny + Pose 两个 ControlNet 同时控制，观察冲突如何解决
3. 训一个 LoRA（10 张图，500 步），生成对比图

### 进阶
4. 阅读 IP-Adapter 论文，理解它与 ControlNet 的设计差异
5. 实现 LoRA + ControlNet 组合推理（注意训练数据来源不同）

---

## §9 FAQ

**Q1：为什么 ControlNet 的副本不直接训练新 UNet？**

A：复用 SD 的特征提取能力。副本的 encoder 与原 UNet encoder 看到的是相同 noisy latent，特征是对齐的。

---

**Q2：LoRA rank 怎么选？**

A：经验值：
- 简单 object（一只特定猫）：r = 4
- 风格（艺术家画风）：r = 8 - 16
- 复杂 concept（独特构图风格）：r = 16 - 32

不要盲目用 r=128 这种大值——大概率过拟合，而且生成时 LoRA 占主导，原 SD 的能力被吃掉。

---

**Q3：能不能"训一次 LoRA，所有 SD 版本通用"？**

A：**不行**。LoRA 与 base model 绑定：
- SD 1.5 的 LoRA 不能直接用在 SDXL（维度都不同）
- 同一 SD 1.5 的不同 fine-tune（如 DreamShaper）上 LoRA 效果会变
- 这是 LoRA 生态碎片化的根本原因

---

## §10 参考资源

- Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models*, ICCV 2023（ControlNet）
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- Ruiz et al., *DreamBooth*, CVPR 2023
- Ye et al., *IP-Adapter*, 2023
- HuggingFace `peft` 库源码
- Stable Diffusion WebUI 的 controlnet 插件

---

> 下一讲（L11）我们离开 SD/UNet 时代，转向**Diffusion Transformer**——Sora 与 SD 3 的架构底座。
