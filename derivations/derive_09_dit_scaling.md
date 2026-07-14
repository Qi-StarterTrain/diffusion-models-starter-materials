# Derivation 09: DiT 的设计选择与 Scaling 分析

> 对应 L11 内容。本手稿分析：
> - AdaLN-Zero 为什么收敛快（vs cross-attention）
> - DiT 的 scaling law（参数与 Gflops vs FID）
> - 与 LLM scaling 的对比

---

## §1 LayerNorm 与 AdaLN

### 1.1 标准 LayerNorm

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

其中 $\gamma, \beta \in \mathbb{R}^D$ 是可学参数。

### 1.2 AdaLN（Adaptive LayerNorm）

让 $\gamma, \beta$ 依赖于 condition $c$：
$$\gamma(c) = W_\gamma c + b_\gamma$$
$$\beta(c) = W_\beta c + b_\beta$$

广泛应用：StyleGAN、BigGAN 都用 AdaIN/AdaBN。在 DiT 里用 AdaLN。

---

## §2 AdaLN-Zero 设计

DiT 论文的关键创新：**zero-initialized AdaLN**。

### 2.1 增强版 AdaLN

每个 DiT block 内有 attn 与 mlp 两段。对每段引入 6 个标量函数 $\gamma_i, \beta_i, \alpha_i$：

```
# Attention 分支
h = (1 + γ_1) ⊙ LN(x) + β_1   # AdaLN before attn
x = x + α_1 ⊙ Attn(h)          # Scaled residual

# MLP 分支
h = (1 + γ_2) ⊙ LN(x) + β_2   # AdaLN before mlp
x = x + α_2 ⊙ MLP(h)          # Scaled residual
```

$\alpha$ 是**残差缩放因子**。

### 2.2 Zero init

```python
self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
nn.init.zeros_(self.adaLN_modulation.weight)
nn.init.zeros_(self.adaLN_modulation.bias)
```

所有 $\gamma, \beta, \alpha$ 初始为 0。

---

### 2.3 初始等价 identity

代入 $\gamma = 0, \beta = 0, \alpha = 0$：
- $h = (1+0) \cdot \text{LN}(x) + 0 = \text{LN}(x)$
- $x = x + 0 \cdot \text{Attn}(h) = x$

整个 block 变成 identity！每个 layer 初始都是 identity，整个网络初始等价于 identity。

---

### 2.4 为什么 identity 有利

类比 ResNet 的 identity skip：
- 深层网络如果每层都是 identity 起点，**没有梯度爆炸/消失**
- 训练开始时网络等价于浅层，逐层"激活"
- 这是 ResNet、Transformer-Pre-Norm、AdaLN-Zero 共享的设计哲学

---

## §3 与其他条件注入方式对比

DiT 论文系统 ablate 4 种条件注入：

| 方式 | 输入位置 | 表达能力 | 训练难度 |
|------|--------|---------|---------|
| In-context | 序列前缀 | 高 | 高 |
| Cross-attn | 每 layer 内 | 高 | 中 |
| AdaLN | 调制 LN | 中 | 低 |
| AdaLN-Zero | + zero init | 中 | **极低** |

实验 FID（ImageNet 256）：

| 方式 | FID |
|------|-----|
| In-context | 9.62 |
| Cross-attn | 7.59 |
| AdaLN | 6.81 |
| **AdaLN-Zero** | **5.55** |

为什么 AdaLN-Zero 最好？
- 训练稳定 → 收敛更深
- 调制 LN 的"开关"比 cross-attn 的"内容"更适合 class label（信息量小但 categorical）
- Zero init 让训练在合理初值附近

---

## §4 DiT Scaling Law

### 4.1 实验设置

DiT 论文训了 4 个尺寸：
- DiT-S/2: 33M params, 24M ops/img
- DiT-B/2: 130M params, 100M ops/img
- DiT-L/2: 458M params, 320M ops/img
- DiT-XL/2: 675M params, 491M ops/img

在 ImageNet 256 上训练相同的 7M images。

### 4.2 Scaling 曲线

| Model | FID (no CFG) | FID (w/ CFG) |
|-------|--------------|--------------|
| DiT-S/2 | 68.4 | 11.2 |
| DiT-B/2 | 43.5 | 5.5 |
| DiT-L/2 | 23.0 | 3.5 |
| DiT-XL/2 | 19.5 | **2.27** |

**关键观察**：
1. FID 与参数量呈幂律下降（power law）
2. CFG 一直有效，不饱和
3. 从 XL/2 到下一倍尺寸，**没看到平台期**

### 4.3 Compute-optimal

定义 "FLOPs spent" = params × training images。

DiT 论文的图 6 显示：
- 在固定 compute budget 下，更大模型 + 少训练步 > 小模型 + 多训练步
- 这与 LLM 的 Chinchilla scaling law **相反**！

---

## §5 与 LLM Scaling 对比

### 5.1 LLM (Chinchilla, Hoffmann 2022)

最优 compute 分配：
$$\text{data tokens} \approx 20 \times \text{params}$$

例：7B 模型应训 140B tokens。

### 5.2 DiT

DiT 实验显示：在 ImageNet 上，**更大模型 + 同数据** > 小模型 + 同数据。

这是因为：
- ImageNet 数据相对小（1.3M images），不是 "data-rich" regime
- 在 data-rich regime（如 SD 用 LAION-5B），可能也走 Chinchilla scaling

### 5.3 SD 3 的 scaling 实验

Esser et al. 2024 (SD 3) 训了 4 个尺寸：
- 800M, 2B, 4B, **8B**

观察：8B 持续优于 4B，scaling law 未饱和。

→ **在 data-rich text-to-image 上，DiT 的 scaling 与 LLM 类似**。

---

## §6 计算量分析

### 6.1 Attention 复杂度

DiT-XL/2 在 256×256 → 16×16 = 256 tokens。
Attention: $O(N^2 d) = O(256^2 \cdot 1152) = 7.5 \times 10^7$ ops。

但 Sora 在 60s × 1080p：
$N = (60 \cdot 30 \cdot 540 \cdot 960) / (4 \cdot 16 \cdot 16) \approx 9 \times 10^5$

Attention: $O(9 \times 10^5)^2 \cdot d \approx 10^{12+}$ ops/layer！**完全不可行**。

→ Sora 必然用 sparse / window attention。

---

### 6.2 FFN 复杂度

每个 token 的 FFN: $O(d^2)$（用 GEGLU 等 hidden expansion）。
N tokens 总共 $O(N d^2)$。

在 large model 下 FFN 主导，不是 attention。

### 6.3 与 UNet 的对比

UNet：spatial conv，复杂度 $O(N \cdot k^2 \cdot c^2)$
- $k$ kernel size（如 3）
- $c$ channels

在小 N、大 c 下 UNet 更高效；大 N、小 c 下 DiT 更高效。

→ 高分辨率 (大 N) 倾向于 UNet，大模型 (大 c) 倾向于 DiT。

---

## §7 Patch size 选择

### 7.1 Trade-off

- Smaller patch → 更多 tokens → 更精细但更贵
- DiT 默认 $p=2$ (在 SD latent 32×32 上 → 16×16=256 tokens)

### 7.2 数学

设 latent shape $L \times L$，patch size $p$。
- N tokens = $(L/p)^2$
- Attention compute: $O(N^2) = O(L^4 / p^4)$

把 $p$ 从 2 变 1：compute × 16。**性价比通常不划算**。

### 7.3 SDXL / SD 3 选择

SD 3 用更大的 latent (64×64) + patch=2 → 32×32=1024 tokens。
- 更精细
- Compute 增 16 倍

通过 8B params 抵消单 token compute 的下降。

---

## §8 与 UNet 的混合架构

### 8.1 UViT (Bao 2022)

把 UNet 的 conv block 全换成 Transformer block，保留 skip connection。
- 实证：比 UNet 好，比 DiT 略差
- 但训练比 DiT 稳

### 8.2 Hunyuan-DiT (Tencent 2024)

DiT + 一些 conv stem → 处理高分辨率时的低层特征。
- 实证：在 prompt-image alignment 上比纯 DiT 好

### 8.3 趋势

完全 conv-free 的 DiT 是未来。但短期内 hybrid 路线在工程上有优势。

---

## §9 实验性自查

### 9.1 Identity init 验证

实验：让 DiT 模型在训练 0 步时 forward，看输出。
预期：$\text{output} \approx \text{noisy input}$（identity）。

代码：
```python
model.eval()
x = torch.randn(2, 4, 32, 32)
t = torch.zeros(2)
y = torch.zeros(2, dtype=torch.long)  # class 0
out = model(x, t, y)
print((out - x).abs().mean())  # 应该接近 0
```

---

### 9.2 AdaLN-Zero gradient norm

预期：训练开始时，AdaLN 参数的 gradient norm 应该**非零**（信号通过 LN 调制传回来）。

如果 grad norm 一直 0（除了 LN），说明 zero init 太"死"了 → 学不动。

---

## §10 自查题

1. 推导 AdaLN-Zero 在 init 时让 block 退化为 identity
2. 计算 DiT-XL/2 推理 1 张 256×256 图的总 FLOPs
3. 比较 DiT 与 UNet 在 1024×1024 上的计算量
4. 解释为什么 DiT 的 scaling law 与 LLM 类似

---

## §11 参考

- Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023
- Hoffmann et al., *Chinchilla* (Compute-optimal scaling), 2022
- Esser et al., *SD 3*, 2024
- Bao et al., *UViT*, 2022
- HunYuanDiT (Tencent), 2024
