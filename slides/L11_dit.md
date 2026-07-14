# L11: Diffusion Transformer (DiT)（W10）

> **目标**：理解为什么 Transformer 成为新一代扩散模型的主干（SD 3, Sora, Flux）
> **前置**：Vision Transformer, L09 (LDM)
> **核心问题**：UNet 是扩散模型的"宿命"吗？

---

## §1 课程定位

L05-L09：所有扩散模型 = UNet + 加噪 + 去噪。
本讲：颠覆这个等式。

**DiT (Peebles & Xie, ICCV 2023)** 证明：把 UNet 全换成 Transformer，扩散模型在 class-conditional ImageNet 上 SOTA。

之后：
- **SD 3** 用 MMDiT（带 text 流的 DiT）
- **Sora** 用 DiT for video
- **Flux** 也用 MMDiT 变种

UNet 时代正在结束。

---

## §2 为什么 UNet 走到尽头

### 2.1 UNet 的设计假设

UNet 适合扩散的核心理由：
1. **多尺度**：encoder-decoder 处理不同 frequency
2. **跳跃连接**：保留细节
3. **空间归纳偏置**：卷积参数效率高

### 2.2 这些假设在大模型时代变成枷锁

| 假设 | 问题 |
|------|------|
| 多尺度 | Transformer 通过 attention 自学多尺度 |
| 跳跃连接 | 信息在 attention 里全连接 |
| 空间归纳偏置 | 大数据下不需要 inductive bias |
| ResBlock 容量有限 | 难扩到 GPT-3 那种规模 |
| FFN 在 UNet 里很短 | 但 LLM 经验显示 FFN 越深越好 |

**关键洞察**：DiT 论文表明，扩散模型的 scaling law 与 LLM 类似——参数越多越好。这要求架构更**同质化**（homogeneous），而不是 UNet 这种"先收缩再扩张"。

---

## §3 DiT 架构

### 3.1 整体流程

```
1. Patchify：z (B, 4, 32, 32) → tokens (B, 256, D)
2. Transformer blocks × N
3. Linear head: (B, 256, D) → (B, 256, 4 * patch²)
4. Unpatchify: (B, 256, 4*p²) → (B, 4, 32, 32)
```

完全 Transformer，没有卷积。

---

### 3.2 Patchify

Latent $z \in \mathbb{R}^{4 \times 32 \times 32}$（SD VAE 输出），patch size $p = 2$：

```python
# z 是 (B, 4, 32, 32)
# 拆成 patches: (B, 16×16, 4×2×2) = (B, 256, 16)
patches = z.unfold(2, p, p).unfold(3, p, p).reshape(B, 4*p*p, -1).transpose(1, 2)
tokens = linear_proj(patches)  # (B, 256, D)
```

**$p$ 选 1 vs 2 vs 4**：
- $p=1$：256×256 patches（计算量爆炸）
- $p=2$：16×16 = 256 patches（DiT 默认）
- $p=4$：8×8 = 64 patches（损失细节）

---

### 3.3 Position Embedding

Sinusoidal 2D position embedding，加到 patch embedding：
```python
pos_emb = get_2d_sincos_pos_embed(D, grid_size=16)  # (256, D)
tokens = tokens + pos_emb
```

---

### 3.4 DiT Block

每个 block 含 self-attention + FFN（标准 transformer）+ **条件注入**。

DiT 论文比较 4 种条件注入方式：

| 方式 | 描述 | FID（ImageNet 256） |
|------|------|---------------------|
| In-context | 把 t, y 当 token 拼到 sequence 前 | 9.6 |
| Cross-attention | 像 SD：用条件做 cross-attn | 7.6 |
| AdaLN | 用条件预测 LN 的 γ, β | 6.8 |
| **AdaLN-Zero** | AdaLN 但 γ 初始化为 0 | **5.5** |

—— **AdaLN-Zero 胜出**。

---

### 3.5 AdaLN-Zero（必看）

```python
class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim)

        # 关键：6 个 scale + shift（attn 前 LN、attn 后 residual、mlp 前 LN、mlp 后 residual）
        self.adaLN = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaLN.weight)  # ← Zero init!
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, c):
        # c 是条件 embedding (B, cond_dim)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN(c).chunk(6, dim=-1)

        # Attn 分支
        h = modulate(self.norm1(x), gamma1, beta1)   # = (1+γ) * LN(x) + β
        x = x + alpha1.unsqueeze(1) * self.attn(h)   # alpha 控制残差强度

        # MLP 分支
        h = modulate(self.norm2(x), gamma2, beta2)
        x = x + alpha2.unsqueeze(1) * self.mlp(h)
        return x
```

**关键设计**：
- 6 个调制（gamma, beta, alpha for attn 与 mlp）
- adaLN 投影**初始化为 0** → 所有调制初始 0 → block 退化为 identity

**初始等价 identity** 让 DiT 像 ResNet 一样易训。

---

### 3.6 条件 c 的构造

$c$ 是 time + class embedding 之和：

```python
t_emb = timestep_embedding(t, D)  # sinusoidal
y_emb = label_embedding(y, D)      # learned
c = t_emb + y_emb                  # 简单相加
```

For text-conditional（SD 3）：text embedding 通过 cross-attn 注入；time + global text token → AdaLN。

---

## §4 Scaling Properties

DiT 论文最重要的 Figure 6：参数量 vs FID。

| Model | Params | Gflops/sample | FID (ImageNet 256, CFG) |
|-------|--------|---------------|-------------------------|
| DiT-S/2 | 33M | 1.4 | 11.2 |
| DiT-B/2 | 130M | 5.6 | 5.5 |
| DiT-L/2 | 458M | 19.7 | 3.5 |
| **DiT-XL/2** | **675M** | **29.1** | **2.27** |

—— Scaling law 严格成立，2.27 是当时 ImageNet 256 SOTA。

**关键启示**：从 DiT-S 到 DiT-XL 参数增 20×，FID 从 11 降到 2.3 ——**没有看到平台期**。这是 SD 3 (8B) 和 Sora 敢于继续 scale 的依据。

---

## §5 MMDiT（SD 3 用）

SD 3 (Esser et al., 2024) 把 DiT 扩展为 **Multi-Modal DiT**。

### 5.1 核心差异

DiT：text 进 AdaLN（只影响 LN 的 γ, β）
MMDiT：text 与 image 在**同一序列**做 self-attention（双流并行）

```
       Image tokens:   [img_1, img_2, ..., img_256]  (256 tokens)
       Text tokens:    [txt_1, txt_2, ..., txt_77]   (77 tokens)
       Joint sequence: concat → 333 tokens
       Attention: full self-attn between all 333
```

但两个流**用不同的 weights**（image stream 的 W_q, W_k, W_v 与 text stream 不同）。

---

### 5.2 为什么这么设计

直觉：在 cross-attn 里 text 是"second-class citizen"（只作为 K, V，不作为 Q）。MMDiT 让 text 也能 query image。

实验：MMDiT 在 GenEval、T2I-CompBench 等 prompt 跟随基准上比纯 cross-attn 强。

---

## §6 DiT 训练细节

### 6.1 训练目标

DiT 用 **EDM-style** loss（不是 simplified DDPM loss）：
- 输入归一化（preconditioning）
- $\sigma$ schedule（不是 $t$ schedule）
- $D_\theta$ prediction（不是 $\epsilon$）

详见 paper note 09 (EDM)。

---

### 6.2 类别 dropout

DiT 用 class-conditional CFG，所以训练时 **10% 类别替换为 NULL**（与 CFG 一致）。

---

### 6.3 数据流水线

ImageNet 256 训练：
- 数据：1.3M images
- Patch size 2，所以每图 256 tokens
- Batch size: 256（small）→ 1024（XL）
- 训练步数：400K（这是论文规模；SD 3 类规模需要 1M+ 步）

---

## §7 工程对比：UNet vs DiT

| 维度 | UNet (SD 1.5) | DiT |
|------|---------------|-----|
| 参数 | 860M | 33M-675M（更灵活） |
| Inductive bias | 强（卷积、多尺度） | 弱（attention）|
| 高分辨率 | encoder/decoder 设计依赖 | 直接加 token 数（不变架构） |
| 视频 | 难（时序依赖、多帧） | 简单（时间维度变成 token 维度） |
| 训练成本 | 标准 | 大 model 下 attention $O(N^2)$ 贵 |
| 推理 | 标准 | 同上 |
| LoRA 兼容 | 成熟 | 较新但可行 |
| 社区生态 | 海量 | 正在追赶（SD 3 后） |

---

## §8 DiT 的"显然"扩展

### 8.1 Video DiT（Sora 思路）

时间是另一个 token 维度：

```
Frames × H × W patches → (T * H * W / p²) tokens
3D attention: spatial + temporal
```

详见 L14, L15。

---

### 8.2 多模态 DiT（生图 + 生视频 + 生 3D）

把不同模态都 tokenize 成统一序列：
- 图像：spatial patches
- 视频：spatio-temporal patches
- 3D：voxel patches or NeRF features
- Action：trajectory tokens

DiT 框架原生支持——只需调整 patchify 与 unpatchify。

---

### 8.3 DiT for VLA / Embodied

未来方向：
- Vision (image tokens) + Language (text tokens) + Proprioception (state tokens) + Action (diffusion target tokens)
- 全部进 DiT，统一架构
- $\pi_0$（Physical Intelligence）部分采用此思路

详见 L17。

---

## §9 课后任务

### 必做
1. 跑通 nb08（DiT block 实现 + AdaLN-Zero 验证）
2. 比较 UNet 和 DiT 在同参数下的 FID（用预训练 model 测）
3. 读 DiT paper §4 + §5（最重要部分）

### 进阶
4. 实现 video DiT 雏形：2D patchify + 3D attention
5. 读 MMDiT 论文 §3，理解 dual stream attention 的设计

---

## §10 FAQ

**Q1：DiT 比 UNet 一定好吗？**

A：**不绝对**。
- 数据少时（10K- 量级）：UNet 因 inductive bias 更高效
- 数据多时（>10M）：DiT 因架构容量天花板高更优
- 工程上：UNet 生态成熟，DiT 是未来但 SD 1.5 仍有海量场景

---

**Q2：AdaLN 比 cross-attn 真的更好？**

A：**对 class-conditional 是的**，对 text-conditional **不一定**。
- DiT 用 class condition (1000 类) → 信息量小 → AdaLN 足够
- SD 用 text (77 tokens) → 信息量大 → 需要 cross-attn 才能精细控制
- MMDiT 是折中：用 self-attn 替代 cross-attn，每个 text token 都能"看" image

---

**Q3：DiT 的 patch size 越小越好吗？**

A：不一定。$p = 1$ 让序列长度从 256 → 1024，attention $O(N^2)$ 变 16 倍。计算量限制下 $p = 2$ 最划算。

---

## §11 参考资源

- Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023（**DiT 原论文**）
- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*, 2024（SD 3 / MMDiT）
- OpenAI, *Video generation models as world simulators* (Sora technical report, 2024)
- HuggingFace `diffusers` 库的 `DiTPipeline` 与 `StableDiffusion3Pipeline`

---

> 下一讲（L12）我们将看到一个**更激进的颠覆**：抛弃 SDE，直接用 ODE 学一个"向量场"——Flow Matching。
