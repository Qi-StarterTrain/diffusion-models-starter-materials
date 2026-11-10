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
- 单张图 tensor 形状 (3, 512, 512)，包含 **262,144 个空间像素、786,432 个标量分量**。
- U-Net 的单次前向，是在某个时间步对一张带噪图像做一次去噪预测，不是完成整张图的生成。
- 计算量取决于通道数、层数、attention 配置等，不能仅由输入尺寸确定。

以普通卷积为例，若一次乘加按 2 FLOPs 计，一层卷积约需：

```math
F_{\mathrm{conv}}\approx 2HW\,C_{\mathrm{in}}C_{\mathrm{out}}k^2
```

在 512×512 分辨率上，一个 128→128 通道的 3×3 卷积就约需 **77.3 GFLOPs**。网络内部有许多层，并非只处理 RGB 三个通道，因此大型像素空间 U-Net 的开销可能达到数 TFLOPs。

**量级示例**：假设某个指定网络单图单次前向为 5 TFLOPs，1000 步采样且每步仅调用一次，则总计 5 PFLOPs；这不是所有 U-Net 的固定开销。标准双分支 CFG 还需计算条件与参考分支。

FLOPs 表示运算总量，TFLOP/s 表示计算速率。实际生成耗时还取决于精度、显存带宽和硬件利用率，不能据此直接断言“A100 上需要 15 秒”。

**核心问题**：人眼能区分的图像，其感知信息量远小于像素数。

> **🔑 LDM 的洞察**：在 perceptually equivalent 但维度更低的 latent space 做扩散，可以大幅降低成本。

---

### 1.2 数字感受

```
原始像素空间：512×512×3 = 786,432 维
↓ 编码（8x 下采样）
Latent 空间：64×64×4 = 16,384 维     （减少 ~48x）
```

这里的 **48 倍是标量维度压缩比，不等于计算量或耗时加速比**。

空间位置从 512² 减至 64²，减少了 **64 倍**。在中间通道数和卷积配置不变时，卷积计算量主要随空间面积变化；attention、模型宽度和 VAE 开销还会影响总成本。较小的特征图显著降低去噪开销，是 SD 能在消费级显卡上运行的重要原因。

---

### 1.3 一个提醒：这是 2022 年的结论

LDM 在 2022 年展示了感知压缩对扩散训练与采样效率的价值，后续许多模型沿用了 latent 路线。但**不要把它记成"像素空间是被解决掉的旧问题"**。

LDM 降低了反复去噪的空间分辨率，代价是引入了一个自编码器——这笔代价 §8 会展开。少步生成时，一次 VAE 解码的相对开销会上升；2026 年的研究也重新探索了从 latent 预训练迁移到 pixel-space 生成的路线（见 [Jiang et al.](https://arxiv.org/abs/2608.16887) 及补充阅读 [Pixel Space vs Latent Space：latent 不是终点](../supplementary/pixel-vs-latent-space.md)）。

本讲先把 LDM 这条主线讲透，但请带着一个问题往下读：**latent space 到底买到了什么，又付出了什么？**

---

## §2 LDM 总体架构

### 2.1 三阶段流程

以下以 **Stable Diffusion 的 KL 路线**为例；原始 LDM 还研究了 VQ 正则化路线（见 §3.1）。DDIM 50 步只是可选采样配置。

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
│  x → Encoder(冻结) → 采样并缩放 → z_0                    │
│  z_0 → 加噪 → z_t                                        │
│  z_t → U-Net (条件 c) → 预测 noise                       │
│  Loss = MSE                                              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Stage 3: 推理                                            │
│                                                          │
│  z_T ~ N(0, I) → DDIM 50 step → z_0 → 逆缩放            │
│                                      → Decoder → image │
└─────────────────────────────────────────────────────────┘
```

---

### 2.2 关键决策点

1. **VAE 与 Diffusion 分离训练**：解耦设计、可独立优化
2. **感知压缩目标**：用感知损失 + 对抗损失保留视觉细节，缓解仅用像素重建损失容易产生的模糊；不保证无损重建
3. **Latent 维度选择**：SD 1.x 的典型形状为 (4, 64, 64)，对应 512×512 图像和空间下采样因子 f=8；并非所有 LDM 都采用此配置
4. **U-Net 在 latent 空间运行**：保留编码器—解码器与 skip connection 骨架，文本条件通过 cross-attention 注入

---

## §3 VAE 的特殊设计

### 3.1 离散 latent 与连续 latent

**Latent（潜变量）**是编码器提取的图像压缩表示：图像 → Encoder → latent → Decoder → 重建图像。

“离散 / 连续”描述的是**表示的取值方式**，不是特征图的空间位置是否离散。

**① 连续 latent：每个位置是可以连续变化的实数向量**

以 Stable Diffusion 的 KL 正则化自编码器（常称 KL-VAE）为例，编码器输出均值与方差，定义条件分布：

```math
q_\phi(z\mid x)=\mathcal N\!\left(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\right)
```

通过重参数化采样得到 latent：

```math
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
```

例如，512×512 图像可编码为 (4, 64, 64) 的特征图，每个位置存一个四维向量：

```text
z[i, j] = [0.21, -1.37, 0.08, 0.64]
```

这些分量可以连续变化，不需要从有限候选集合中选择。训练中的 KL 项约束条件分布接近标准正态，但不保证所有图像的 latent 整体严格服从标准正态。

**② 离散 latent：每个位置从有限码本中选择一个向量**

VQ-VAE 的 VQ 指 **Vector Quantization（向量量化）**。模型学习一个含 K 个向量的码本：

```math
\mathcal E=\{e_1,\ldots,e_K\},\qquad e_k\in\mathbb R^d
```

编码器先输出连续向量 h，再用最近的码本向量替换它：

```math
k^*=\arg\min_k\|h_{i,j}-e_k\|_2^2,\qquad z_{q,i,j}=e_{k^*}
```

```text
编码器输出 h[i, j] → 查找最近码字 → 编号 317 → 码本向量 e_317
```

所以，同一个量化结果既能表示为**整数编号网格**，也能表示为**查码本得到的向量特征图**。“离散”指可选码字数量有限，码本向量本身仍由实数构成。

| 对比 | 连续 latent | 离散 latent |
|---|---|---|
| 直观类比 | 在调色盘上自由调色 | 从固定色卡中选颜色 |
| 每个位置的取值 | 实数向量 | 有限码本中的某个向量 |
| 是否需要量化 | 不需要 | 需要最近邻码本选择 |
| 典型方法 | KL 正则化自编码器 | VQ-VAE / VQGAN |

**③ 离散码本是否与高斯扩散不兼容？**

要区分扩散作用于**整数编号**还是**实数向量**。标准高斯加噪为：

```math
z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad\epsilon\sim\mathcal N(0,I)
```

- **直接对编号加噪**：编号 317 加噪变成 317.42，并没有合理的码本含义；编号之间的数值距离也不代表视觉相似度。若要在离散编号上生成，需要设计适合离散状态的生成过程。
- **对实数向量加噪**：量化前的编码器输出、查码本后的向量，都可以在数学上加入高斯噪声；带噪向量不必落在码本中。具体采用哪种表示，需要说明模型设计。

**原始 LDM 的 VQ 路线采用量化前的连续表示训练扩散模型，并把量化操作归入解码器。** 因此，VQ 正则化并不迫使扩散模型直接处理离散编号。

> **🔑 关键**：LDM 论文同时研究了 KL-reg 与 VQ-reg 两条路线，并非“VQ-VAE 失败后只剩 KL-VAE”。本讲后续以 Stable Diffusion 使用的 KL 正则化连续 latent 为主。

参考：[LDM 原论文 §3.1 与附录 G](https://arxiv.org/html/2112.10752v2#S3.SS1)（附录 G 说明了扩散训练时量化层的位置）。

---

### 3.2 Stable Diffusion 用的 KL-VAE

下面只示意编码器输出均值与对数方差的接口，省略具体模块构造与重参数化；SD 1.x 编码器的通道倍率为 [1, 2, 4, 4]。

```python
class Encoder(nn.Module):
    """8x 下采样，输出连续 latent."""
    def __init__(self, ch=128, z_channels=4):
        # 4-level conv encoder
        ...

    def forward(self, x):
        h = self.conv_blocks(x)  # (B, ch*4, H/8, W/8)
        mean = self.mean_proj(h)
        logvar = self.logvar_proj(h)
        return mean, logvar  # 4 channels each

class Decoder(nn.Module):
    """与 Encoder 镜像."""
    def forward(self, z):
        return self.conv_blocks(z)
```

**训练 loss（自编码器侧的示意目标）**：这里 E(x) 简写了后验采样得到的 latent；GAN 项表示生成器侧损失，判别器另外训练。实际实现还会设置各项权重或自适应对抗权重。
```math
\mathcal{L}_{\text{VAE}} = \underbrace{\| x - D(E(x)) \|^2}_{\text{recon}} + \underbrace{\mathcal{L}_{\text{LPIPS}}}_{\text{perceptual}} + \lambda_{\text{adv}} \cdot \underbrace{\mathcal{L}_{\text{GAN}}}_{\text{adversarial}} + \lambda_{\text{KL}} \cdot \underbrace{D_{\text{KL}}}_{\text{regularization}}
```

四个部分：
- **Recon**：像素 MSE/L1，保证基本重构
- **LPIPS**（Zhang 2018）：基于预训练 VGG 特征的距离，保证感知相似
- **GAN**（PatchGAN discriminator）：高频细节，避免模糊
- **KL**：弱正则化（$`\lambda_{\text{KL}} \approx 10^{-6}`$），让 latent 接近高斯但不过度规整

> **🔑 关键**：KL 权重**故意设得极小**，以优先保留重建质量。它仍采用 VAE 式的概率编码与 KL 正则化；生成时由扩散模型学习 latent 的分布，而不是直接从自编码器的标准正态先验采样后解码。

---

### 3.3 Latent 的统计性质

弱 KL 正则化并不保证编码后的 latent 方差为 1。LDM 的 KL 路线会从数据估计 latent 的标准差，并引入 **scaling factor** 调整其幅度：

```math
s\approx\frac{1}{\operatorname{Std}(z_{\mathrm{raw}})},\qquad
z_0=s\,z_{\mathrm{raw}}
```

目的在于让信号尺度与噪声调度匹配，控制扩散过程的**信噪比**；乘一个常数不会把任意分布变成高斯分布。

```python
scaling_factor = vae.config.scaling_factor  # SD 1.5 为 0.18215

# x 已归一化到 [-1, 1]
z_raw = vae.encode(x).latent_dist.sample()
z_0 = z_raw * scaling_factor               # 训练扩散前缩放
x_recon = vae.decode(z_0 / scaling_factor).sample  # 解码前逆缩放
```

0.18215 是 SD 1.x 使用的模型相关常数，对应未缩放 latent 标准差约 5.49，不能推广到所有 VAE。原始 LDM 附录 G 描述了用首批数据估计尺度的办法，不需要遍历整个训练集。

> **🔑 区分两个分布**：干净 latent 的数据分布不必是高斯；高斯的是前向过程加入的噪声，以及充分加噪后的近似终端分布。

参考：[LDM 附录 G](https://arxiv.org/html/2112.10752v2#A7)、[Diffusers AutoencoderKL 接口](https://huggingface.co/docs/diffusers/api/models/autoencoderkl)。

---

## §4 Stable Diffusion 完整架构

### 4.1 模块构成

以下展示 SD 1.5 在 512×512 文生图中的核心组件；tokenizer、scheduler 和 latent 逆缩放也是流水线的一部分。短 prompt 会补齐到 77 个 token 位置（含特殊 token），不是必须写满 77 个词。

```
┌─────────────────────────────────────────────────┐
│ CLIP Text Encoder (ViT-L/14, frozen)             │
│ Input:  prompt → tokenizer → 77 token positions  │
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
│ VAE Decoder                                     │
│ Input:  z_0 / scaling_factor (1, 4, 64, 64)      │
│ Output: image (1, 3, 512, 512)                   │
└─────────────────────────────────────────────────┘
```

---

### 4.2 U-Net 细节

以下固定讨论 **SD 1.5 U-Net**，避免与 SDXL 的 attention 布局混淆：

1. **文本条件**：在选定分辨率的 ResNet 块后接 Spatial Transformer，内部包含 self-attention 和文本 cross-attention；并非每个 ResBlock 后都有
2. **模型规模**：约 860M 参数；DDPM U-Net 的规模随具体配置变化，不用一个固定参数量概括
3. **分辨率布局**：下采样路径在 64×64、32×32、16×16 使用 cross-attention，8×8 下采样块不含 attention；**8×8 的 mid block 含 attention**，上采样路径在对应层恢复 attention

伪代码：
```python
class SDUNet(nn.Module):
    def __init__(self):
        # 配置：ch_mult = [1, 2, 4, 4]，base_ch = 320
        # 在 64, 32, 16, 8 四个分辨率
        # 下采样 cross-attention 在 64, 32, 16；mid block 在 8
        ...

    def forward(self, z_t, t, text_emb):
        h = self.init_conv(z_t)
        t_emb = self.time_emb(t)

        for block in self.down_blocks:
            h = block.resnet(h, t_emb)
            if block.has_cross_attention:
                h = block.spatial_transformer(h, text_emb)
            ...
```

参考：[SD 1.5 U-Net 配置](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/unet/config.json)。

---

### 4.3 Cross-Attention 的实现

```python
import math
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8):
        super().__init__()
        inner_dim = query_dim
        assert inner_dim % heads == 0
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
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
        return self.to_out(out)
```

**attention 核心计算复杂度**：设 N=HW、总特征维度为 d，cross-attention 为 $`O(NTd)`$，self-attention 为 $`O(N^2d)`$。只看注意力矩阵大小，分别为 N×T 与 N×N。当 T=77 且 N 很大时，cross-attention 的矩阵更小；SD 的 Transformer 块中仍包含 self-attention，不能据此忽略它的成本。上例省略了外层归一化、残差与前馈网络。

---

## §5 SD 推理完整流程（代码级解剖）

下面固定使用 **SD 1.5 + DDIM（eta=0）**，省略加载和图像保存。先加载 `StableDiffusionPipeline`，将 VAE、U-Net、文本编码器放在同一设备上并设为 eval，再传入 `pipe`；这里不调用 `pipe(prompt)`，也不使用 CPU offload。示例只支持单张 512×512 文生图。

```python
import torch
from diffusers import DDIMScheduler

@torch.no_grad()
def stable_diffusion_inference(
    pipe,
    prompt: str,
    negative_prompt: str = "",
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = None,
):
    tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder
    unet, vae = pipe.unet, pipe.vae
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    device, dtype = unet.device, unet.dtype

    # ─────────── 1. Tokenize（正、负提示词） ───────────
    text_input = tokenizer(prompt, padding="max_length", max_length=77,
                           truncation=True, return_tensors="pt")

    uncond_input = tokenizer(negative_prompt, padding="max_length",
                              max_length=77, truncation=True,
                              return_tensors="pt")

    # ─────────── 2. 文本编码 ───────────
    cond_emb = text_encoder(text_input.input_ids.to(device))[0]
    uncond_emb = text_encoder(uncond_input.input_ids.to(device))[0]

    # concat for batched CFG
    text_emb = torch.cat([uncond_emb, cond_emb], dim=0).to(dtype)
    # (2, 77, 768)；negative_prompt 非空时，uncond 是负条件参考分支

    # ─────────── 3. 初始化 latent ───────────
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    # 在 CPU 采样再转到模型设备，兼容 CUDA / MPS / CPU
    z = torch.randn((1, 4, 64, 64), generator=generator)
    z = z.to(device=device, dtype=dtype)
    scheduler.set_timesteps(num_steps, device=device)
    z = z * scheduler.init_noise_sigma

    # ─────────── 4. 采样循环（DDIM + CFG） ───────────
    for t in scheduler.timesteps:
        # CFG: 同时跑 cond + uncond
        z_in = torch.cat([z, z], dim=0)  # (2, 4, 64, 64)
        z_in = scheduler.scale_model_input(z_in, t)
        eps = unet(z_in, t, encoder_hidden_states=text_emb).sample

        # 拆分并 CFG 外推
        eps_uncond, eps_cond = eps.chunk(2)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # DDIM 步进
        z = scheduler.step(eps, t, z, eta=0.0).prev_sample

    # ─────────── 5. 逆缩放与 VAE 解码 ───────────
    z = (z / vae.config.scaling_factor).to(dtype=vae.dtype)
    image = vae.decode(z).sample

    # ─────────── 6. 图像后处理 ───────────
    image = (image / 2 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]

    return image
```

CFG 的两个分支合并为 batch=2 后，一次 Python 调用可完成两份预测，但 FLOPs 仍近似为两份，峰值激活显存也会上升。更换 scheduler 时，需要保留相应的初始化、输入缩放和步进约定，不能只替换循环中的名字。

参考：[Diffusers 手写推理流水线](https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline)、[DDIMScheduler 接口](https://huggingface.co/docs/diffusers/api/schedulers/ddim)。

---

## §6 SD 工程实战要点

### 6.1 显存优化

显存需求取决于模型版本、精度、batch、CFG、attention 后端与 offload 策略。**6GB 不是 SD 1.5 的固定最低要求**。下表说明各技术的作用，不给脱离基准环境的固定加速或节省百分比：

| 技术 | 主要作用 | 代价或限制 |
|------|------|------|
| fp16 / bf16 | 相比 fp32，浮点权重存储约减半；总显存不一定减半 | 需要硬件支持，部分计算可能保留 fp32 |
| Attention slicing | 分块计算注意力，降低中间张量峰值 | 可能变慢；已有 SDPA / xFormers 时通常不应再叠加 |
| VAE tiling | 分块解码，降低高分辨率解码的峰值显存 | 可能增加耗时并产生局部色调差异 |
| Model CPU offload | 按组件迁移权重，减少同时驻留 GPU 的模型 | 增加 CPU—GPU 传输 |
| Sequential CPU offload | 更细粒度迁移，进一步节省显存 | 传输频繁，通常更慢 |
| PyTorch SDPA / xFormers | 使用优化 attention 后端，降低注意力显存与耗时 | 实际内核取决于硬件、dtype 和张量形状，不保证固定加速 |

FlashAttention 类内核的硬件要求随实现和版本而异，不能概括为“需要 A100+”。参考：[Diffusers 显存优化](https://huggingface.co/docs/diffusers/optimization/memory)。

### 6.2 batch 推理

batch 推理通常能提高吞吐量，但不保证吞吐量或延迟线性变化。模型权重在 batch 内共享，激活显存随 batch 增长：
```python
# 4 张图 batch
z = torch.randn((4, 4, 64, 64))
# 双分支 CFG 时，U-Net 的有效输入 batch 为 8
```

可粗略理解为 $`M_{\text{peak}}\approx M_{\text{weights}}+M_{\text{activations}}(B)+M_{\text{workspace}}`$。因此不能把单图总显存乘以 4 来估计四图显存，也不存在通用的“24GB 最优 batch=4–8”；应在目标配置上测量峰值显存与每秒出图数。

---

## §7 SD 的几个版本

以下是与本讲有关的历史节点，不是截至当前的完整版本清单；FLUX 是独立模型家族，列在这里用于对比。

| 版本 | 时间 | 关键改进 |
|------|------|---------|
| SD 1.4 | 2022 Aug | 初版发布，CLIP-L/14 text encoder |
| SD 1.5 | 2022 Oct | 延续 SD 1.x 架构与训练；本讲的代码对象 |
| SD 2.0 | 2022 Nov | OpenCLIP ViT-H 文本编码器；配置与权重不能直接混用 SD 1.x |
| SD 2.1 | 2022 Dec | SD 2.x 的后续训练版本；需区分具体 checkpoint 的分辨率与预测参数化 |
| SDXL | 2023 Jul | base U-Net 约 2.6B 参数，双文本编码器；可选 refiner 继续去噪 |
| SD 3 | 2024 | MMDiT 架构（L11 / W10），Rectified Flow / Flow Matching（L12 / W11） |
| SD 3.5 | 2024 | SD 3 改进版 |
| FLUX.1 | 2024 | Black Forest Labs 的独立系列；Transformer + Rectified Flow |

> **🔑 延续与变化**：这些模型延续了“先感知压缩，再在 latent 上生成”的路线，但 U-Net→Transformer、噪声预测→流匹配、文本编码器和 latent 通道数都有实质变化；§4–§5 的 SD 1.5 实现不能原样套到 SD 3 / FLUX。

参考：[SDXL 论文 Table 1](https://arxiv.org/html/2307.01952v1#S2.SS1)、[SD 3 论文](https://arxiv.org/html/2403.03206v1)、[FLUX.1-dev 模型卡](https://huggingface.co/black-forest-labs/FLUX.1-dev)。

---

## §8 LDM 的限制

### 8.1 VAE 重构损失

VAE 不是无损压缩，x → z → 重建图像的全过程可能损失细节。特别是：
- **文字**：常被模糊
- **小目标**：可能丢失
- **高频纹理**：被平均化

**缓解**：SD 3 将 latent 通道数从 4 增至 16，提高表示容量和重建质量，但也增加生成模型需要处理的信息量。这不保证文字或小目标错误全部消失，因为生成网络本身也会产生错误。参考：[SD 3 论文 §5.2.1 与附录 B.2](https://arxiv.org/html/2403.03206v1)。

### 8.2 感知压缩不保证保留任务所需的全部信息

“感知等价”是设计目标，不是严格保证。训练损失偏好的视觉相似，不一定等同于文字识别、几何测量或科学成像需要的精确性；一旦编码阶段丢失了信息，下游生成网络就难以恢复。

latent 保留空间结构、整体分布不是严格高斯，都不构成高斯扩散的数学障碍。其训练难度取决于表示、网络和任务；不能据此断言它比像素空间更难学。

### 8.3 两阶段训练带来的不对齐

自编码器主要优化重建与感知质量，去噪网络则学习 latent 分布，这两个目标并非联合优化。更换 VAE、数据域或缩放约定，都可能改变去噪网络面对的表示分布，需要验证兼容性。

端到端联合优化是一种可能的研究方向，但不能将 SD 3 作为现成例子：其论文使用预训练自编码器，并预计算图像 latent 来训练生成模型。参考：[SD 3 论文附录 B.2 与 §5.3.1](https://arxiv.org/html/2403.03206v1)。

### 8.4 推理时的 decode 开销

VAE decode 是一次**固定成本**：不管采样多少步，最后都要跑一遍 decoder。

```math
T_{\text{total}}\approx T_{\text{text}}+\sum_{i=1}^{N_{\text{calls}}}T_{\text{denoise},i}+T_{\text{decode}}+T_{\text{other}}
```

这里 $`N_{\text{calls}}`$ 指实际去噪网络调用次数。对 §5 的 DDIM，每步一次 batch=2 调用，单次耗时已包含 CFG 两分支；换采样器后，步数与调用次数可能不同。报告 NFE 时应说明是否把 CFG 两分支分别计数。

- **50 步采样**时，重复去噪通常占更大的耗时比例，但解码是否可忽略要实测；
- **减少至 4 步**时，若其他配置不变，解码的绝对耗时近似不变，而其占比会上升。蒸馏模型还可能同时改变网络和 CFG，不能只按步数推算速度（L13 / W12）。

因此，不能把 latent 空间单次去噪的节省直接当作端到端加速；低步数下尤其需要计入固定开销。像素空间是否更快，还取决于模型、patch 大小、步数和目标质量，详见 [Pixel Space vs Latent Space：latent 不是终点](../supplementary/pixel-vs-latent-space.md)。

> §10 进阶档第 6 题就是让你自己量一量这个比例随 NFE 怎么变。

---

## §9 本讲核心要点

1. **LDM** 在压缩表示上反复去噪；SD 1.x 示例中标量维度减少 48 倍，不等于固定的计算加速比
2. **自编码器设计**：原始 LDM 支持 KL / VQ 正则化；SD 1.x 使用连续 latent + 感知损失 + GAN + 弱 KL
3. **Scaling factor** 调整 latent 幅度与信噪比，不负责把分布变成高斯；0.18215 是 SD 1.x 的配置
4. **SD 1.5 核心组件**：VAE + U-Net + CLIP 文本编码器 + 可替换的 scheduler（本讲以 DDIM 为例）
5. **Cross-attention** 是文本条件注入的关键
6. **CFG** 可通过 batch concat 合并两分支调用，计算量仍近似为两份
7. **工程优化**（低精度、优化 attention 后端、按需 offload）降低显存需求，具体收益需实测
8. **latent 不是终点**：VAE 带来重构上限与一次固定的 decode 开销，低 NFE 下这笔成本的占比显著上升——见 [Pixel Space vs Latent Space](../supplementary/pixel-vs-latent-space.md)

---

## §10 课后任务（Project 3 启动）

### 必做（Project 3 基础档）

1. **代码任务**：用 `diffusers` 库做 SD 推理深度解剖
   - 加载 SD 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`)
   - 手动实现 6 阶段推理（tokenize、encode、init noise、loop、decode、postprocess），与 §5 对应
   - 不允许调 `pipe(prompt)`，必须每步手写
   - 提交 `01_inference_walkthrough.ipynb`

2. **参数扫描**：固定 prompt 和 seed，扫描以下参数对生成图的影响
   - CFG scale: [1, 3, 7.5, 15, 25]
   - DDIM steps: [10, 20, 50, 100]
   - 不同 sampler: DDIM, Euler-A, DPM++ 2M
   提交 `02_parameter_sweep.ipynb` + 对比网格图

3. **Reading note**：LDM 论文 §3-§4 + SDXL technical report 第一部分

### 进阶档

4. **VAE 解剖**：可视化 SD 1.5 VAE 编码后的 4 个 latent channels，比较统计量与空间结构；不要预设每个通道对应一种固定、可命名的语义

5. **Cross-attention 可视化**：提取并可视化 cross-attention map（哪个词 attend 到哪个图像区域）

6. **decode 开销占比测量**：固定 prompt、seed、分辨率、CFG 和精度，用 DDIM 分别运行 50 / 20 / 4 步，记录实际 U-Net 调用次数及有效 batch。预热后重复测量，CUDA 上计时前后用 `torch.cuda.synchronize()` 同步；MPS 使用对应同步接口。分别计时文本编码、denoising loop、decode 与完整流水线，区分 decode / (denoise + decode) 和 decode / 端到端耗时。低步数的原版 SD 1.5 可能质量下降，此实验只测开销构成，不等价于复现蒸馏效果。代码骨架与思考题见 [Pixel Space vs Latent Space §9](../supplementary/pixel-vs-latent-space.md)；该骨架只测 denoise + decode，按本题要求补齐其他计时。

### 挑战档

7. **SDXL inference**：自己组装 SDXL 双模型（base + refiner）推理流水线

---

## §11 推荐进一步阅读

| 资源 | 重点 |
|------|------|
| LDM 论文 | §3 必读 |
| SDXL technical report | 双模型架构 |
| [推导手稿 derive_03_ddpm_loss.md](../derivations/derive_03_ddpm_loss.md) | 同样的高斯扩散训练目标可用于连续 latent |
| diffusers 源码 | 工业级实现参考 |
| HuggingFace blog *The Annotated Diffusion Model* | 配合代码学习 |
| Jiang et al., *An Empirical Study of Training Pixel-Space Text-to-Image Diffusion Models* (arXiv:2608.16887) | 反方向视角：latent 的代价与 pixel-space 的回归。需要 L11/L12/L13 基础，建议 W13 之后读，导读见 [补充材料](../supplementary/pixel-vs-latent-space.md) |

---

## §12 常见问题

**Q: SD 已经生成了 latent，为什么还要 decoder？**

A: SD 1.5 生成的 latent 是 4 通道的学习表示，不是 RGB 图像；需要 Decoder 将它映射为像素。可以逐通道可视化 latent 来观察空间结构，但这不等于正确解码后的图像。

**Q: 为什么 SD 训练时不直接在像素上做 diffusion？**

A: 主要动机是减少反复去噪时的计算与显存开销，同时利用自编码器提取的感知表示。具体训练成本取决于网络、数据、硬件和训练步数，不能给所有配置统一套用一个 GPU-days 数字。

2026 年 [Jiang et al. 的实验](https://arxiv.org/abs/2608.16887)观察到，直接在像素空间做大规模预训练收敛更慢；作者随后采用 latent 预训练 → pixel-space 后训练的路线，在其测试配置中实现了质量相当或更好的结果，并报告 3.18–4.75× 端到端推理加速。这是特定训练配方和架构下的结果，不能理解为给原版 SD 1.5 去掉 VAE 就会自动加速。

这提供了“latent 预训练、像素空间生成”的可行路线，尚不能概括成所有模型的最优选择。展开见 [Pixel Space vs Latent Space](../supplementary/pixel-vs-latent-space.md)。

**Q: SD 的 0.18215 scaling factor 是怎么算出来的？**

A: 可理解为用于归一化 latent 幅度的标准差倒数，见 §3.3。使用已有模型时应读取 `vae.config.scaling_factor` 并保持训练时约定；自行训练新的自编码器与 LDM 时，再用代表性数据估计尺度。只换推理 prompt 或数据，不应随意重算已有模型的缩放值。

**Q: SD 为什么用 CLIP text encoder 而不是 BERT 或 T5？**

A: SD 1.x 选用了图文对比学习预训练的 CLIP 文本编码器，其表示适合提供视觉相关的文本条件。但图文预训练不是必要条件：纯文本预训练的 T5 也能提供有效条件，[Imagen 论文](https://arxiv.org/abs/2205.11487)就展示了这种路线；SD 3 进一步组合 CLIP 与 T5。效果取决于语言理解能力、模型规模和训练设计，不能简单断言 T5 一定更难训练或更差。

**Q: 干净 latent 不是高斯，会违反 DDPM 的假设吗？**

A: 不会。DDPM 不要求干净数据服从高斯，像素图像本来也不是高斯。对连续 latent，同样有：

```math
z_T=\sqrt{\bar\alpha_T}\,z_0+\sqrt{1-\bar\alpha_T}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I)
```

当残留信号相对于噪声足够小时，终端分布才近似标准高斯。有限噪声日程可能带来终端分布与采样起点的偏差，这是像素扩散和 latent 扩散都可能遇到的问题，不能归因于弱 KL。scaling factor 调节信号幅度，也不等价于保证干净 latent 高斯化。

**Q: SD 能不能做高分辨率（如 1024×1024）？**

A: SD 1.5 主要面向 512×512，直接生成 1024×1024 可能出现重复主体等分辨率泛化问题。SDXL 支持以 1024×1024 为代表的更高分辨率及多种宽高比。也可先生成，再用超分模型或分阶段加噪去噪放大；具体支持范围取决于 checkpoint 与推理配置。

---

> **下一讲预告**：L10 进入 P2 阶段的第一讲——**ControlNet 与 LoRA**。前面九讲都在"从零造一个扩散模型"，从这里开始换个思路：在动不了的大模型上**做加法**。这也是工业界真实的工作流。再往后 L11 才是 **DiT (Diffusion Transformer)**——继 U-Net 之后的新一代架构，Sora、SD 3、Flux 都基于它。
