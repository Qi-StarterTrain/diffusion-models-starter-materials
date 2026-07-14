# Derivation 10: 视频扩散模型的 Attention Factorization

> 对应 L14-L15 内容。本手稿推导：
> - 时空 attention 的不同 factorization 策略
> - 各自的计算复杂度与表达能力
> - Sora 可能用的优化策略

---

## §1 设置

输入：video latent $z \in \mathbb{R}^{T \times H \times W \times C}$

patch 后 tokens：$z \in \mathbb{R}^{N \times d}$，其中 $N = T \cdot H \cdot W / (p_t p_h p_w)$

**关心两件事**：
1. 计算复杂度（$O(N^2)$ 不可行时怎么办）
2. 表达能力（能否建模长程时空依赖）

---

## §2 全 attention（baseline）

### 2.1 公式

$$\text{Attn}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d}) V$$

### 2.2 复杂度

- Time: $O(N^2 d)$
- Memory: $O(N^2)$（存 attention matrix）

在视频上 $N$ 极大：
- 32 frames × 32×32 latent → N = 32768
- $N^2 = 10^9$ → 完全不可行

---

## §3 Factorization 策略 1: Spatial + Temporal

### 3.1 算法

```
z (T, HW, d)
  ↓ spatial attention (HW tokens within each frame)
z (T, HW, d)
  ↓ temporal attention (T tokens at each spatial position)
z (T, HW, d)
```

### 3.2 复杂度

- Spatial attn: $T$ 个独立 attention, 每个 $O((HW)^2 d) = O(N^2 d / T^2)$。总 $O(N^2 d / T)$。
- Temporal attn: $HW$ 个独立, 每个 $O(T^2 d) = O(N^2 d / (HW)^2)$。总 $O(N^2 d / (HW))$。

总：$O(N^2 d / T) + O(N^2 d / (HW)) \approx O(N^2 d / \min(T, HW))$

例：$T = 32, HW = 1024 \to$ 速度提升 32 倍。

### 3.3 表达能力损失

无法直接建模"frame 5 的位置 (3,7) 与 frame 8 的位置 (12, 4)"的依赖（必须经过 spatial→temporal 两步）。

但实际上：spatial 与 temporal 交替几层后，全局依赖能复现。

实证：在 Video Diffusion (Ho 2022)、Imagen Video 中工作良好。

---

## §4 Factorization 2: Spatio-temporal Window

### 4.1 算法

把 token 划分成 spatio-temporal **windows** $W_{t,h,w}$（如 $4 \times 8 \times 8$）。

```
z (T, H, W, d)
  ↓ 分窗
windows: list of (4 * 8 * 8, d) blocks
  ↓ 每窗内做 full attention
↓ shift window (类比 Swin Transformer)
↓ 再做一层
```

### 4.2 复杂度

设 window size $w$。每个 window 计算 $O(w^2 d)$。总 windows: $N/w$。
$$O(N/w \cdot w^2 d) = O(N w d)$$

例：$w = 256 \to O(N \cdot 256 \cdot d)$，比 $O(N^2 d)$ 在大 N 下快得多。

### 4.3 Shift window

类比 Swin Transformer：奇偶层窗口偏移半个 window，让相邻窗口的 token 建立联系。

经验：4-8 层后，整个 N tokens 都能"互通"。

---

## §5 Factorization 3: Hybrid（Sora 推测）

### 5.1 设计原则

混合 spatial + temporal + window：

```
Layer 1-N/4: spatial attention only (建立 frame-internal 表征)
Layer N/4 - N/2: temporal attention (建立 frame-cross 关系)
Layer N/2 - 3N/4: window attention (refine local spatio-temporal)
Layer 3N/4 - N: sparse full attention (global)
```

### 5.2 Sora 的疑似策略

业界推测 Sora 用：
- **Long-range global attention** in early layers（看大局）
- **Local window** in middle layers（refine）
- 配合 **flash attention** 等工程优化

但无官方确认。

---

## §6 Sparse Attention（更激进）

### 6.1 BigBird / Longformer 思路

每个 token 不是 attend 所有其他 token，而是 attend：
1. **Local**（附近 window）
2. **Global tokens**（几个固定的"hub" tokens）
3. **Random**（随机几个）

复杂度：$O(N \log N)$ 或 $O(N \sqrt{N})$。

### 6.2 在 video 上的应用

- Local：时空相邻 tokens
- Global：帧的"中央 token"
- Random：模型自己决定

实验：CogVideoX、Lumiere 都用类似思路。

---

## §7 Linear Attention

### 7.1 思路

把 softmax 用 kernel 替代，让 attention 变成 $O(N)$。

$$\text{softmax}(QK^T) V \approx \phi(Q) (\phi(K)^T V)$$

其中 $\phi$ 是 kernel feature map（如 ELU+1）。

### 7.2 Performer / Linformer

复杂度：$O(N d^2)$（取决于 $d$ 与 sequence dim 的关系）。

### 7.3 实践

视频领域用得少——performance gap 比图像/文本大。可能因为视频需要"sharp" attention（具体位置匹配），linear 的 soft 不够。

---

## §8 Spatio-Temporal Compression

### 8.1 思路

不在 attention 上想办法，而是**降低 N**：

- 用 **3D VAE** 替代 2D VAE：把 $(T, H, W) \to (T/2, H/8, W/8)$
- N 直接减半 → attention 计算 4 倍便宜

### 8.2 应用

Open-Sora 2.0、Mochi 1 都用 3D VAE：
- $f_t = 4$（时间下采样 4 倍）
- $f_s = 8$（空间下采样 8 倍）

### 8.3 缺点

- 3D VAE 难训（比 2D VAE 数据需求高）
- 重构 video 比重构 image 难

---

## §9 KV Cache & Long Video Generation

### 9.1 自回归视频生成

```
generate frames 1-16
  ↓ KV cache the 16 frames' features
generate frames 17-32 (using cache)
  ↓ slide window: keep frames 9-32 in cache
generate frames 33-48
...
```

每次新生成一段，复用之前帧的 KV。

### 9.2 Memory budget

KV cache size: $O(N \cdot d \cdot L)$（L 是 layer 数）。

实际：16 frames × 32×32 patches × 1152 dim × 28 layers × fp16 ≈ 1 GB。

100 frames cache → 6.25 GB（可控）。

### 9.3 Quality drift

KV cache 让模型"看不到未来"，但**真正限制**是：早期生成的帧质量错误会传到后续。

解决：定期"重启"——重新生成一段，把错误纠正。

---

## §10 实证比较

各种 factorization 的实证质量（FVD on UCF-101，越低越好）：

| 方法 | FVD | TFLOPs/sample |
|------|-----|----------------|
| Full 3D attn | **80** | 1500 |
| Spatial + Temporal | 95 | 80 |
| Window | 110 | 60 |
| Sparse | 100 | 70 |
| Linear | 130 | 40 |

Full attn 最优但极贵；spatial+temporal 是性价比之王。

---

## §11 数学注记：Positional Embedding

视频需要 3D positional embedding。

### 11.1 加法（DiT-like）

$$z_n = z_n + \text{PE}(t_n, h_n, w_n)$$

3D sinusoidal：
$$\text{PE}(t, h, w) = \text{concat}[\text{PE}_t(t), \text{PE}_h(h), \text{PE}_w(w)]$$

### 11.2 RoPE 3D

应用 rotary embedding 到 Q, K：
$$Q_n \to \text{RoPE}(t_n, h_n, w_n) \cdot Q_n$$

实证：RoPE 3D 在长视频上比 sinusoidal 好（外推性）。

---

## §12 自查题

1. 推导 spatial+temporal factorization 的计算复杂度
2. 设计一个 "global token" 的 hybrid attention，写伪代码
3. 解释 KV cache 在长视频生成中如何工作
4. 思考：为什么 linear attention 在视频上 work 不好？

---

## §13 参考

- Ho et al., *Video Diffusion Models*, 2022
- Bertasius et al., *TimeSformer*, ICML 2021（factorized 3D attention 经典）
- Liu et al., *Swin Transformer*, ICCV 2021
- Beltagy et al., *Longformer*, 2020
- Choromanski et al., *Performer*, ICLR 2021
