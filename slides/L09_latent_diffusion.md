# Lecture 09｜Latent Diffusion Model 与 Stable Diffusion

> **本讲目标**
> - 理解 Latent Diffusion 的核心动机与设计
> - 完整解剖 Stable Diffusion 的工程架构
> - 理解 VAE 编码器、U-Net、文本编码器、采样器如何协作
> - 掌握 SD 推理流水线的每一步
> - 为 W9（LoRA、ControlNet、微调）做架构准备

> **对应论文**：Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022

---

## §1 像素空间扩散的瓶颈

### 1.1 计算成本

到 W7 为止，我们的扩散都在**像素空间**进行。

对于 512×512 RGB 图像：
- 单张图 tensor 形状 (3, 512, 512) = 786,432 个像素
- 一个标准 U-Net 单次前向需要 ~5 TFLOPs
- 1000 步采样 = 5 PFLOPs
- 在 A100 上单张图生成需要 ~15 秒

**核心问题**：人眼能区分的图像，其感知信息量远小于像素数。

> **🔑 LDM 的洞察**：在 perceptually equivalent 但维度更低的 latent space 做扩散，可以大幅降低成本。

---

### 1.2 数字感受

```
原始像素空间：512×512×3 = 786,432 维
↓ 编码（8x 下采样）
Latent 空间：64×64×4 = 16,384 维     （减少 ~48x）
```

48 倍维度压缩 ≈ 48 倍计算量降低（粗略估计）。这是为什么 SD 能在普通显卡上运行。

---

## §2 LDM 总体架构

### 2.1 三阶段流程

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1: 训练 VAE（一次性，独立）                         │
│                                                          │
│  x_image  ──[Encoder]──>  z_latent                       │
│  z_latent ──[Decoder]──>  x_recon                        │
│                                                          │
│  Loss = recon + perceptual + adversarial + KL            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Stage 2: 训练 Latent Diffusion（在 z 空间）              │
│                                                          │
│  x → Encoder(冻结) → z_0                                 │
│  z_0 → 加噪 → z_t                                        │
│  z_t → U-Net (条件 c) → 预测 noise                       │
│  Loss = MSE                                              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Stage 3: 推理                                            │
│                                                          │
│  z_T ~ N(0, I) → DDIM 50 step → z_0 → Decoder → image   │
└─────────────────────────────────────────────────────────┘
```

---

### 2.2 关键决策点

1. **VAE 与 Diffusion 分离训练**：解耦设计、可独立优化
2. **VAE 设计为"perceptually equivalent"压缩**：用对抗损失 + 感知损失，避免高斯 VAE 的模糊问题
3. **Latent 维度选择**：典型 (4, 64, 64) for 512×512 图像，下采样因子 f=8
4. **U-Net 在 latent 空间运行**：架构与像素空间 DDPM 几乎相同，但输入是 latent

---

## §3 VAE 的特殊设计

### 3.1 VQ-VAE 的失败教训

LDM 论文实验了 VQ-VAE（离散 latent）和 KL-VAE（连续 latent），最终选用 KL-VAE。

VQ-VAE 的问题：
- 离散 token 让扩散模型必须处理离散变量
- 与连续 noise 不兼容

---

### 3.2 LDM 用的 KL-VAE

```python
class Encoder(nn.Module):
    """8x 下采样，输出连续 latent."""
    def __init__(self, ch=128, z_channels=4):
        # 4-level conv encoder
        ...

    def forward(self, x):
        h = self.conv_blocks(x)  # (B, ch*8, H/8, W/8)
        mean = self.mean_proj(h)
        logvar = self.logvar_proj(h)
        return mean, logvar  # 4 channels each

class Decoder(nn.Module):
    """与 Encoder 镜像."""
    def forward(self, z):
        return self.conv_blocks(z)
```

**训练 loss**：
$$\mathcal{L}_{\text{VAE}} = \underbrace{\| x - D(E(x)) \|^2}_{\text{recon}} + \underbrace{\mathcal{L}_{\text{LPIPS}}}_{\text{perceptual}} + \lambda_{\text{adv}} \cdot \underbrace{\mathcal{L}_{\text{GAN}}}_{\text{adversarial}} + \lambda_{\text{KL}} \cdot \underbrace{D_{\text{KL}}}_{\text{regularization}}$$

四个部分：
- **Recon**：像素 MSE/L1，保证基本重构
- **LPIPS**（Zhang 2018）：基于预训练 VGG 特征的距离，保证感知相似
- **GAN**（PatchGAN discriminator）：高频细节，避免模糊
- **KL**：弱正则化（$\lambda_{\text{KL}} \approx 10^{-6}$），让 latent 接近高斯但不过度规整

> **🔑 关键**：KL 权重**故意设得极小**——这不是真正的 VAE（不需要从 prior 采样），而是一个"轻微正则化的 autoencoder"。

---

### 3.3 Latent 的统计性质

LDM 训完 VAE 后，对训练集中所有 latent 计算 mean/std：
- mean ≈ 0
- std ≈ 1（但不精确）

为了让 latent 接近标准高斯（diffusion 模型的假设），LDM 引入 **scaling factor**：

```python
SCALING_FACTOR = 0.18215  # SD 1.5 经验值

z = encoder(x) * SCALING_FACTOR  # 编码后乘
x = decoder(z / SCALING_FACTOR)  # 解码前除
```

这个 0.18215 的来历：在 LAION 子集上算出的 latent 标准差倒数的近似值。

---

## §4 Stable Diffusion 完整架构

### 4.1 模块构成

SD 1.5 完整组件：

```
┌─────────────────────────────────────────────────┐
│ CLIP Text Encoder (ViT-L/14, frozen)             │
│ Input:  "a cat playing chess" (77 tokens)        │
│ Output: (1, 77, 768) text embeddings             │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼ (注入 cross-attention)
┌─────────────────────────────────────────────────┐
│ U-Net (~860M params)                             │
│ Input:  z_t (1, 4, 64, 64) + t + text_emb        │
│ Output: pred_noise (1, 4, 64, 64)                │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼ (DDIM 采样 50 步)
┌─────────────────────────────────────────────────┐
│ z_0 (1, 4, 64, 64)                               │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ VAE Decoder (~50M params)                        │
│ Input:  z_0 (1, 4, 64, 64)                       │
│ Output: image (1, 3, 512, 512)                   │
└─────────────────────────────────────────────────┘
```

---

### 4.2 U-Net 细节

SD U-Net 的关键改进（相对于 DDPM U-Net）：

1. **接受 cross-attention 输入**：每个 ResBlock 后插入 cross-attention 层
2. **更大模型**：860M 参数 vs DDPM 的 ~50M
3. **多分辨率 attention**：不只在 16x16，在 32x32、16x16、8x8 都加 attention

伪代码：
```python
class SDUNet(nn.Module):
    def __init__(self):
        # 配置：ch_mult = [1, 2, 4, 4]，base_ch = 320
        # 在 64, 32, 16, 8 四个分辨率
        # cross-attention 在 32, 16, 8（不在 64，省计算）
        ...

    def forward(self, z_t, t, text_emb):
        h = self.init_conv(z_t)
        t_emb = self.time_emb(t)

        for block in self.down_blocks:
            h = block.resnet(h, t_emb)
            h = block.cross_attn(h, text_emb)  # 关键！
            ...
```

---

### 4.3 Cross-Attention 的实现

```python
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8):
        super().__init__()
        inner_dim = query_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.heads = heads

    def forward(self, x, context):
        # x: (B, HW, query_dim) image features
        # context: (B, T, context_dim) text embeddings (T=77)
        B, HW, _ = x.shape
        q = self.to_q(x)             # (B, HW, dim)
        k = self.to_k(context)       # (B, T, dim)
        v = self.to_v(context)       # (B, T, dim)

        # Multi-head reshape
        q, k, v = [t.reshape(B, -1, self.heads, t.shape[-1] // self.heads)
                       .transpose(1, 2) for t in (q, k, v)]
        # (B, heads, HW or T, head_dim)

        attn = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        attn = attn.softmax(dim=-1)
        out = attn @ v               # (B, heads, HW, head_dim)
        out = out.transpose(1, 2).reshape(B, HW, -1)
        return out
```

**计算复杂度**：$O(HW \cdot T)$，远低于 self-attention 的 $O(HW^2)$。

---

## §5 SD 推理完整流程（代码级解剖）

下面是简化版 SD 推理代码，理解每一行：

```python
@torch.no_grad()
def stable_diffusion_inference(
    prompt: str,
    negative_prompt: str = "",
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = None,
):
    # ─────────── 1. 文本编码 ───────────
    text_input = tokenizer(prompt, padding="max_length", max_length=77,
                           return_tensors="pt")
    cond_emb = text_encoder(text_input.input_ids)[0]   # (1, 77, 768)

    uncond_input = tokenizer(negative_prompt, padding="max_length",
                              max_length=77, return_tensors="pt")
    uncond_emb = text_encoder(uncond_input.input_ids)[0]

    # concat for batched CFG
    text_emb = torch.cat([uncond_emb, cond_emb], dim=0)  # (2, 77, 768)

    # ─────────── 2. 初始化 latent ───────────
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    z = torch.randn((1, 4, 64, 64), generator=generator)  # initial noise

    # ─────────── 3. 采样循环（DDIM） ───────────
    scheduler.set_timesteps(num_steps)
    for t in scheduler.timesteps:
        # CFG: 同时跑 cond + uncond
        z_in = torch.cat([z, z], dim=0)  # (2, 4, 64, 64)
        eps = unet(z_in, t, text_emb).sample

        # 拆分并 CFG 外推
        eps_uncond, eps_cond = eps.chunk(2)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # DDIM 步进
        z = scheduler.step(eps, t, z).prev_sample

    # ─────────── 4. VAE 解码 ───────────
    z = z / SCALING_FACTOR
    image = vae.decode(z).sample
    image = (image / 2 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]

    return image
```

---

## §6 SD 工程实战要点

### 6.1 显存优化

512×512 SD inference 至少需要 ~6GB 显存。优化技巧：

| 技术 | 节省 | 代价 |
|------|------|------|
| fp16 | 50% | 微小精度损失 |
| Attention slicing | 20-30% | 微小速度损失 |
| VAE tiling | 大幅节省（高分辨率） | 解码慢 |
| CPU offload | 70%+ | 速度大幅下降 |
| xformers | (速度+25%) | 需要额外依赖 |
| Flash Attention | (速度+30%) | 需要 A100+ |

### 6.2 batch 推理

SD 在 batch 推理时几乎线性 scale，但显存占用线性增长：
```python
# 4 张图 batch
z = torch.randn((4, 4, 64, 64))
# 推理时显存约 6GB × 4 ≈ 24GB（不能简单这么算，但可参考）
```

实际工程中：4-8 batch_size 是 24GB 显存的甜点。

---

## §7 SD 的几个版本

| 版本 | 时间 | 关键改进 |
|------|------|---------|
| SD 1.4 | 2022 Aug | 初版发布，CLIP-L/14 text encoder |
| SD 1.5 | 2022 Oct | 微调，社区主流 |
| SD 2.0 | 2022 Nov | OpenCLIP-H text encoder，**变化大**，社区接受度低 |
| SD 2.1 | 2022 Dec | 数据清洗，质量回升 |
| SDXL | 2023 Jul | 3.5B 参数双模型架构 + refiner |
| SD 3 | 2024 | DiT 架构（W10 内容），Flow Matching |
| SD 3.5 | 2024 | SD 3 改进版 |
| Flux.1 | 2024 | Black Forest Labs，DiT + Flow Matching |

> **🔑 SD 系列的"灵魂"**：在 latent space 做 diffusion 这个核心架构 4 年没变，变的只是细节。

---

## §8 LDM 的限制

### 8.1 VAE 重构损失

VAE 不是无损压缩，z → x 解码会丢失细节。特别是：
- **文字**：常被模糊
- **小目标**：可能丢失
- **高频纹理**：被平均化

**缓解**：SD 3 用更大的 VAE（16 channels 而非 4），减小这个问题。

### 8.2 Latent 不是真正的"perceptually equivalent"

理论上 KL 让 latent 接近高斯，但实际 latent 有强 structure（边缘、平面区分明）。这让 diffusion 学起来比纯像素难。

### 8.3 两阶段训练带来的不对齐

VAE 和 U-Net 分开训练，可能存在数据流不匹配。SD 3 探索了 end-to-end 训练。

---

## §9 本讲核心要点

1. **LDM** 把 diffusion 从像素空间搬到 latent 空间，48x 减少计算
2. **VAE 设计**：连续 latent + LPIPS + GAN + 极弱 KL，避免高斯 VAE 模糊问题
3. **Scaling factor (0.18215)** 让 latent 接近标准高斯
4. **SD = VAE + U-Net + CLIP text encoder + DDIM**
5. **Cross-attention** 是文本条件注入的关键
6. **CFG** 在 SD 中通过 batch concat 高效实现
7. **工程优化**（fp16, attention slicing, xformers）让 SD 可在消费级显卡运行

---

## §10 课后任务（Project 3 启动）

### 必做（Project 3 基础档）

1. **代码任务**：用 `diffusers` 库做 SD 推理深度解剖
   - 加载 SD 1.5 (`runwayml/stable-diffusion-v1-5`)
   - 手动实现 6 阶段推理（tokenize、encode、init noise、loop、decode）
   - 不允许调 `pipe(prompt)`，必须每步手写
   - 提交 `01_inference_walkthrough.ipynb`

2. **参数扫描**：固定 prompt 和 seed，扫描以下参数对生成图的影响
   - CFG scale: [1, 3, 7.5, 15, 25]
   - DDIM steps: [10, 20, 50, 100]
   - 不同 sampler: DDIM, Euler-A, DPM++ 2M
   提交 `02_parameter_sweep.ipynb` + 对比网格图

3. **Reading note**：LDM 论文 §3-§4 + SDXL technical report 第一部分

### 进阶档

4. **VAE 解剖**：可视化 SD VAE 编码后的 4 个 latent channels，理解它们各自学到了什么

5. **Cross-attention 可视化**：提取并可视化 cross-attention map（哪个词 attend 到哪个图像区域）

### 挑战档

6. **SDXL inference**：自己组装 SDXL 双模型（base + refiner）推理流水线

---

## §11 推荐进一步阅读

| 资源 | 重点 |
|------|------|
| LDM 论文 | §3 必读 |
| SDXL technical report | 双模型架构 |
| 推导手稿 derive_03_ddpm_loss.pdf | DDPM 训练目标（latent space 不变） |
| diffusers 源码 | 工业级实现参考 |
| HuggingFace blog *The Annotated Diffusion Model* | 配合代码学习 |

---

## §12 常见问题

**Q: 为什么 SD 不直接用 stable diffusion 模型 latent 来生成？为什么还要 decoder？**

A: Latent 是高维抽象表示（4 channels 编码了图像感知信息），不是图像本身。Decoder 才把它"渲染"成 RGB 像素。Latent 直接显示是无意义的颜色噪声。

**Q: 为什么 SD 训练时不直接在像素上做 diffusion？**

A: 计算开销。在 512×512 像素上训练 LDM 规模的模型需要数千 GPU-days。LDM 的核心贡献就是把这个成本降到几十 GPU-days，让公开训练变得可行。

**Q: SD 的 0.18215 scaling factor 是怎么算出来的？**

A: 论文作者在训练数据子集上算 `1 / latent.std()` ≈ 0.18215。如果你自己训 LDM，要在自己的数据上重新算这个值。

**Q: SD 为什么用 CLIP text encoder 而不是 BERT 或 T5？**

A: CLIP 是对齐图文的，其文本 embedding 自带"视觉概念"信息。BERT/T5 是纯语言模型，没有视觉对齐。Imagen 用 T5 是因为他们的设计强调 deep language understanding，但代价是更难训练。

**Q: latent space diffusion 有什么数学上的问题？**

A: 严格来说，latent 不是真正的高斯先验（VAE 训练只做了弱 KL）。在 latent 上做 diffusion 时，DDPM 假设的"$x_T \sim \mathcal{N}(0,I)$"不严格成立。实践中影响不大，但理论上 LDM 与原始 DDPM 有 mismatch。

**Q: SD 能不能做高分辨率（如 1024×1024）？**

A: SD 1.5 训练于 512×512，直接生成 1024×1024 会有"重复人脸"等问题（OOD）。SDXL 训练于 1024×1024，可以直接生成。或者用 super-resolution diffusion 做 upscale。

---

> **下一讲预告**：L10 我们进入 **DiT (Diffusion Transformer)**——继 U-Net 之后的新一代架构。Sora、SD 3、Flux 都基于 DiT。你将看到 attention is all you need 在 diffusion 上的胜利。
