# L14: Video Diffusion 基础（W13）

> **目标**：从图像扩展到视频，理解时序建模的核心挑战
> **前置**：L09 (LDM), L11 (DiT)
> **核心问题**：视频是"很多帧的图像"还是"另一种数据模态"？

---

## §1 课程定位

P0-P1：图像扩散基础。
本讲：增加**时间维度**。

视频生成的代表作：
- **Video Diffusion Models** (Ho et al., 2022)：开山，基于 UNet 3D
- **Imagen Video** (Ho et al., 2022)：Google 的级联 video diffusion
- **Stable Video Diffusion (SVD)** (Blattmann et al., 2023)：开源 video 模型
- **Sora** (OpenAI, 2024)：DiT-based video，质量飞跃
- **Veo / Movie Gen / Mochi**：后续大模型

> 本讲专注**基础架构与训练策略**，L15 专门讲 Sora。

---

## §2 朴素方法：逐帧独立扩散

最简单做法：用 SD 对每帧独立生成。

**问题**：
- 帧间无关联 → 闪烁、跳变
- 无时序一致性

例：用户输入 "a cat running"
- 朴素：8 帧各自生成不同的猫，颜色、姿态都不同
- 期望：8 帧呈现同一只猫在跑动

---

## §3 关键挑战：时序一致性

视频生成必须解决的问题：

| 一致性维度 | 含义 | 例子 |
|----------|------|------|
| **Identity** | 主体跨帧一致 | 猫的颜色、纹理 |
| **Motion** | 运动平滑 | 不能瞬移 |
| **Lighting** | 光照一致 | 不能突然变天 |
| **Scene** | 场景延续 | 同一房间，不能切场 |
| **Causality** | 因果合理 | 杯子掉下来应该破 |

最难的是**长时序**（>5 秒）的 motion 与 causality。

---

## §4 经典架构：3D UNet（Ho et al., 2022）

### 4.1 整体思路

把 2D UNet 的卷积/attention 扩展到 3D（含时间维）。

输入 shape：(B, C, T, H, W)，T 是帧数。

### 4.2 三个关键模块

**1. 时空卷积**
```
2D conv: (C, H, W) → (C, H', W')
3D conv: (C, T, H, W) → (C, T', H', W')
```

但 3D conv 参数多、计算贵。实际用 **factorized conv**：
```
spatial conv (H, W) + temporal conv (T)
```

**2. 时空 attention**

类似：spatial self-attn + temporal self-attn 分别做。

```python
# Pseudo
x = x + spatial_attn(x)   # H × W 内 attend
x = x + temporal_attn(x)  # T 内 attend (每个 spatial location 独立)
```

**3. 时间位置编码**

把 frame index $t \in [0, T-1]$ 用 sinusoidal embedding 加到 feature。

---

### 4.3 Imagen Video 的级联策略

Imagen Video 把 video 生成拆成多个 stage：

```
Base model:    16 frames @ 24×40   →  text-conditional
SR temporal:   16 → 64 frames      →  插值新帧
SR spatial:    24×40 → 640×1280     →  超分
```

每个 stage 独立训练，串行推理。
- 单个 stage 简单（专注一个任务）
- 级联让总质量提升（但推理慢）

---

## §5 现代方法：DiT-based Video

Sora 等放弃 UNet，用 DiT-3D。

### 5.1 Patchify 3D

输入：(B, C, T, H, W)
patch size: (p_t, p_h, p_w)
tokens 数：$N = (T/p_t) \cdot (H/p_h) \cdot (W/p_w)$

例：$T=32, H=W=64, p_t=p_h=p_w=2 \to N = 16 \cdot 32 \cdot 32 = 16384$ tokens。

> 比 2D 多得多！这是 video DiT 的算力瓶颈。

### 5.2 完整 self-attention

DiT-3D 让所有 token 间做 attention：
- 一个 frame 的 token 既看自己 frame 也看其他 frame 的所有 token
- 时空建模自然，但 $O(N^2)$ 巨贵

Sora 的工程优化：window attention + sparse attention。

---

## §6 训练策略

### 6.1 Joint image-video training

Imagen Video 与 SVD 都用：
- 训练时随机选 batch（一会儿全图、一会儿全视频）
- 全图时 T=1
- 共享参数

好处：
- 图像数据远多于视频（10B vs 100M）
- 图像训练给视频模型注入 spatial prior
- 减小 video-only 训练时的过拟合

---

### 6.2 Curriculum

典型 curriculum：
1. **Stage 1**：256×256 图像，1B images，1M steps
2. **Stage 2**：256×256 短视频（4 帧），100M videos，500K steps
3. **Stage 3**：高分辨率高 FPS（64 帧 1024×576）

直接用最终配置训会发散。

---

### 6.3 Image-to-Video (I2V)

很多模型支持"给一张图，生成视频"：
- 输入：1 张图 $x_0^{(0)}$（已知）+ 后续 T-1 帧（噪声）
- 模型：UNet 接受 mask（哪些帧已知）
- 训练：随机 mask 一些帧作为已知

SVD 默认 I2V 模式：用户给一张图，模型生成 14 帧动画。

---

## §7 数据策略

### 7.1 Caption 难

视频 caption 比图像 caption 难得多：
- 要描述动作、变化、因果
- 长 caption 帧间一致性才好

Sora 的解决：用 GPT-4V 给每个 video 生成详细 caption（"DALL-E 3 trick" 的视频版）。

### 7.2 数据来源

| 数据集 | 规模 | 来源 |
|--------|------|------|
| WebVid-10M | 10M | Shutterstock |
| HD-VG-130M | 130M | 自爬 |
| Sora 训练 | 上千 PB | 未公开（疑似 YouTube）|

视频数据**版权敏感**——这是 Sora 不开源的部分原因。

---

## §8 评估指标

### 8.1 视频 FID（FVD）
Fréchet Video Distance：用 I3D（pretrained on Kinetics）抽特征，计算与真实视频的 FD。

### 8.2 CLIP-S
对每帧用 CLIP 算 image-text similarity，取平均。

### 8.3 一致性指标
- Frame consistency：相邻帧 CLIP feature 余弦相似度
- Motion smoothness：帧差的二阶导

### 8.4 人类评估
还是最准的——VBench 等标准化人评。

---

## §9 工程实战：用 SVD 推理

```python
from diffusers import StableVideoDiffusionPipeline
import torch

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()  # 显存优化

image = load_image("input.jpg")
frames = pipe(image, num_frames=14, num_inference_steps=25,
              motion_bucket_id=127).frames[0]
```

**显存要求**：SVD 推理 ~10 GB（fp16）。比 SD 大约 3 倍。

---

## §10 视频生成的"硬"挑战

### 10.1 时间长度
SVD: 14 帧（约 1 秒）
Sora: 60 秒
Pika/Runway: 4-10 秒

为什么难？$O(T \cdot H \cdot W)$ attention 在长视频时 OOM。Sora 用 **patches over space-time** 缓解。

### 10.2 物理一致性
扩散模型不"理解"物理：水流方向、重力、碰撞经常错。
Sora 通过海量训练**部分**学到，但仍有失败案例（玻璃杯落地不破）。

### 10.3 复杂动作
- 简单：相机平移、人走路
- 中等：手部动作、抓取
- 困难：复杂运动、特技

---

## §11 与 VLA / Embodied 的连接

### 11.1 World Model 的雏形

视频生成模型 = **隐式世界模型**（隐式预测 next frame）。
具身研究关心的是：
- 给定 action $a_t$，预测 next observation $o_{t+1}$
- 这正是 video conditional generation

代表工作：**Genie** (Bruce et al., 2024, DeepMind)、**WorldDreamer**、**1X World Model**。

详见 L16。

### 11.2 训练数据的"action label"

普通视频没有 action label，但具身研究需要：
- Ego4D（第一人称视频，部分有动作）
- 仿真生成视频（无限数据但 domain gap）
- 机器人示教数据（少但精确）

---

## §12 课后任务

### 必做
1. 用 SVD 生成 5 个不同 input image 的视频，观察 motion 类型
2. 跑 nb11（简化版 video UNet）
3. 阅读 SVD 论文 §3 + §4

### 进阶
4. 实现 image-to-video：用单帧条件训一个小 video diffusion
5. 比较 SVD 与 Pika 在同一 input 上的输出
6. 思考：视频长度从 1s 到 1min 的工程挑战清单

---

## §13 FAQ

**Q1：视频生成不就是"多个图像 + 时间注意力"吗？**

A：**简化版是**，但真要做好涉及：
- Joint image-video training
- 不同分辨率的 curriculum
- Patchify 3D 的设计
- 时序 attention 的稀疏化（不能纯 dense）

工程复杂度远高于"图像 + 时间维"。

---

**Q2：Sora 这么贵，普通研究者怎么做 video?**

A：
- 选小规模问题（短视频 256×256）
- 用 SVD 等开源 backbone fine-tune
- 关注**领域问题**（如机器人、医学、工业），不要追 general video 质量

---

**Q3：视频 vs 多图（4D image）的区别？**

A：4D image（如 NeRF 视角）也有"多个相关视图"，但：
- 视频帧之间是**因果**的（时间）
- 4D 视图之间是**几何**的（视角）

数学上可以统一为"latent code + view/time embedding"，但物理上不同。

---

## §14 参考资源

- Ho et al., *Video Diffusion Models*, 2022（**开山论文**）
- Ho et al., *Imagen Video*, 2022
- Blattmann et al., *Stable Video Diffusion*, 2023
- Bar-Tal et al., *Lumiere*, 2024 (Google space-time UNet)
- Polyak et al., *Movie Gen* (Meta), 2024
- HuggingFace `StableVideoDiffusionPipeline` 源码

---

> 下一讲（L15）我们深入 Sora 的内部——OpenAI 的视频革命。
