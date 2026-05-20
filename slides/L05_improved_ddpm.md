# Lecture 05｜Improved DDPM：Cosine Schedule、Learned Variance 与 Importance Sampling

> **本讲目标**
> - 理解 DDPM 的三个关键工程改进：cosine schedule、learned variance、importance sampling
> - 看懂 NLL（negative log-likelihood）作为评估指标的局限
> - 掌握 hybrid loss 的设计思路
> - 为后续 Score SDE 的统一视角做铺垫

> **对应论文**：Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, ICML 2021

---

## §1 为什么需要"改进"？

L04 我们已经看到 DDPM 训练目标极其简洁：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

DDPM 在 FID 上已经追平甚至超过 GAN。但 Nichol & Dhariwal 指出了**两个未解决的问题**：

1. **NLL 仍然落后**：DDPM 在 negative log-likelihood（NLL）指标上不如最好的 likelihood-based 模型（PixelCNN++ 等）
2. **采样效率低**：1000 步太慢，能否在更少步数下保持质量？

本讲聚焦问题 1（NLL 改进），W6 处理问题 2（DDIM）。

---

### 1.1 NLL vs FID：两种不同的"好"

- **FID**：测量生成分布与真实分布在 Inception 特征空间的距离 → 衡量**视觉质量**
- **NLL（bits per dim）**：测量真实数据在模型下的对数似然 → 衡量**密度估计精度**

> **🔑 关键观察**：DDPM 的 simplified loss 实际上是 ELBO 的"去权重版"。这让 FID 变好，但牺牲了 NLL。

Improved DDPM 的整体哲学：**找到既能保证 FID、又能改进 NLL 的训练方法**。

---

## §2 Cosine Schedule

### 2.1 Linear Schedule 的问题

回顾 L03：
$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$$

可视化 $\bar\alpha_t$ 随 $t$ 的变化：

```
ᾱ_t (linear):
1.0 ──╲
      ╲╲
0.5    ╲╲
        ╲╲___
0.0          ──────────────── (T)
              ↑
              t=200 已经接近 0，信号几乎丢光
```

**问题**：
- 在前 20% 时间步里，信号已经被破坏殆尽
- 后 80% 的时间步在"接近纯噪声"区域，训练信号弱
- 实际损失了大量模型容量

---

### 2.2 Cosine Schedule 的设计

Nichol & Dhariwal 的想法：**让 $\bar\alpha_t$ 在中间区域下降得更平缓**。

$$\bar\alpha_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

其中 $s = 0.008$ 是小偏移量，防止 $t=0$ 处导数为零。

由 $\bar\alpha_t$ 反推 $\beta_t$：
$$\beta_t = 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}$$

**实践**：把 $\beta_t$ clip 到 $(10^{-5}, 0.999)$，防止数值问题。

---

### 2.3 对比可视化

```
ᾱ_t (cosine):
1.0 ──╲
       ╲╲
0.5     ╲╲╲
           ╲╲╲___
0.0              ──╲___ (T)
                   ↑
                   t=800 才接近 0，给了更多有效训练步
```

**直观效果**：
- 中间时间步（t=200~800）有更多"有意义"的加噪状态
- 模型有更多机会学习如何处理"半噪声"图像
- 最终 NLL 提升约 0.04 bits/dim（CIFAR-10）

---

### 2.4 工程实现

```python
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clip(betas, 1e-5, 0.999)
```

> **🔑 实践经验**：
> - 64×64 及以上分辨率：cosine 比 linear 有明显改进
> - 32×32（CIFAR-10）：改进很小，可能在 FID 上反而略差
> - 这是为什么很多 SOTA 论文混合使用：低分辨率用 cosine，高分辨率会调整

---

## §3 Learned Variance（学习方差）

### 3.1 DDPM 固定方差的局限

L04 中我们提到，DDPM 把反向方差固定为 $\sigma_t^2 = \beta_t$ 或 $\sigma_t^2 = \tilde\beta_t$。

但严格 ELBO 推导显示：
- 大 $t$ 区域：最优方差 ≈ $\beta_t$
- 小 $t$ 区域：最优方差 ≈ $\tilde\beta_t$
- 中间区域：在两者之间插值

**固定方差等于做了次优选择**。

---

### 3.2 学习"插值系数"而非方差本身

直接让网络预测 $\Sigma_\theta$ 很难（要保证正定）。Nichol & Dhariwal 提出：**让网络预测一个标量 $v$，在 log 空间插值**：

$$\Sigma_\theta(x_t, t) = \exp\left(v \cdot \log \beta_t + (1 - v) \cdot \log \tilde\beta_t\right)$$

其中 $v = v_\theta(x_t, t) \in [0, 1]$（通过 sigmoid 输出）。

**好处**：
- $v=0$ 时方差为 $\tilde\beta_t$（lower bound）
- $v=1$ 时方差为 $\beta_t$（upper bound）
- 任何 $v$ 都保证方差合理（log-space 插值避免负数）

---

### 3.3 网络架构改动

U-Net 的输出从 $C$ 通道（噪声 $\epsilon$）变成 $2C$ 通道：
- 前 $C$ 通道：预测 $\epsilon$
- 后 $C$ 通道：预测 $v$（每个像素一个，sigmoid 后用作插值）

```python
class UNet(nn.Module):
    def __init__(self, ..., learn_variance=False):
        out_ch = 2 * in_channels if learn_variance else in_channels
        ...
```

---

### 3.4 训练目标：Hybrid Loss

如果只用 simplified loss，方差预测网络拿不到梯度。

**Hybrid loss**：
$$\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{\text{simple}} + \lambda \cdot \mathcal{L}_{\text{vlb}}$$

其中：
- $\mathcal{L}_{\text{simple}}$：噪声预测 MSE（对应 $\epsilon$ 通道）
- $\mathcal{L}_{\text{vlb}}$：完整 ELBO（对应 $v$ 通道）
- $\lambda = 0.001$（小权重）

**关键细节**：对 $v$ 通道**只反传 $\mathcal{L}_{\text{vlb}}$ 的梯度**，不让 $\mathcal{L}_{\text{vlb}}$ 干扰已经好的噪声预测。

```python
def hybrid_loss(model, x0, t, schedule, lambda_vlb=0.001):
    out = model(x_t, t)
    pred_eps, pred_v = out.chunk(2, dim=1)

    L_simple = ((pred_eps - epsilon) ** 2).mean()

    # 对 pred_v 用 detach 的 pred_eps 计算 vlb（隔离梯度）
    L_vlb = compute_vlb(x_t, t, pred_eps.detach(), pred_v, schedule)

    return L_simple + lambda_vlb * L_vlb
```

---

### 3.5 效果

- NLL：CIFAR-10 从 3.70 降到 3.57 bits/dim
- FID：基本不变（这是好事，说明没有牺牲质量换 NLL）

---

## §4 Importance Sampling

### 4.1 不同 $t$ 的训练困难度不同

实测发现：$\mathcal{L}_t$（每个时间步的 loss 贡献）随 $t$ 变化巨大。

```
loss vs t (CIFAR-10):
   t=0     : L ≈ 0.005
   t=100   : L ≈ 0.012
   t=500   : L ≈ 0.020
   t=999   : L ≈ 0.040 ← 训练最困难
```

**均匀采样 $t$ 浪费了大量训练资源**——大部分 batch 落在"容易"的时间步上。

---

### 4.2 重要性采样

让"困难"时间步被更频繁采样：

$$p(t) \propto \sqrt{\mathbb{E}[\mathcal{L}_t^2]}$$

实现：
1. 维护每个 $t$ 的近期 loss 历史（如最近 10 次）
2. 根据 loss 平方均值的开方作为采样权重
3. 在 batch 中按这个分布采样 $t$

```python
class LossAwareSampler:
    def __init__(self, T, history_size=10):
        self.T = T
        self.loss_history = [[] for _ in range(T)]
        self.history_size = history_size

    def sample(self, batch_size):
        weights = self._compute_weights()
        t = torch.multinomial(weights, batch_size, replacement=True)
        return t, 1.0 / (weights[t] * self.T)  # 返回 importance weights

    def update(self, t, losses):
        for t_i, loss_i in zip(t.tolist(), losses.tolist()):
            self.loss_history[t_i].append(loss_i)
            if len(self.loss_history[t_i]) > self.history_size:
                self.loss_history[t_i].pop(0)
```

训练中使用 importance weights 校正 loss：
$$\mathcal{L} = \frac{1}{B} \sum_i w_i \cdot \mathcal{L}_{t_i}$$

---

### 4.3 效果

- 训练收敛速度加快 2-3 倍
- 最终 NLL 进一步改进
- 不影响 FID

---

## §5 三个改进的综合效果

| 改进 | NLL (CIFAR-10, bits/dim) | FID |
|------|--------------------------|-----|
| 原始 DDPM | 3.70 | 3.17 |
| + cosine schedule | 3.66 | 3.21 |
| + learned variance | 3.57 | 3.18 |
| + importance sampling | **3.54** | 3.21 |

**FID 几乎不动**（说明视觉质量没牺牲），**NLL 显著改进**（追平最好的 likelihood 模型）。

---

## §6 一个意外发现：少步采样

Improved DDPM 还展示了一个意外结果：**用 learned variance 训练后，少步采样质量明显更好**。

```
Sampling steps vs FID (CIFAR-10):
   T_inference = 1000:  FID ≈ 3.17  (full)
   T_inference = 100:   FID ≈ 5.5   (DDPM baseline)
   T_inference = 50:    FID ≈ 8.9
   T_inference = 25:    FID ≈ 17.1
```

但 learned variance + 跳步采样：
```
   T_inference = 100:   FID ≈ 4.0   (learned σ)
   T_inference = 50:    FID ≈ 5.5
   T_inference = 25:    FID ≈ 10.2
```

为什么？因为 learned variance 给了模型在"大步长"采样时调整不确定性的能力。

> **🔑 重要预告**：W6 我们将看到 DDIM 用一种完全不同的方式实现少步采样——把整个采样视为 ODE。

---

## §7 模型规模实验

Improved DDPM 还做了一项重要的**scaling 实验**：模型越大效果越好，FID 和 NLL 几乎线性改进。

| Params | FID (CIFAR-10) |
|--------|----------------|
| 21M | 3.21 |
| 35M | 2.94 |
| 51M | 2.83 |

**意义**：扩散模型的 scaling law 成立。这一发现影响了后续 DiT、Stable Diffusion 等工作——**敢于堆参数**。

---

## §8 本讲核心要点

1. **Cosine schedule** 在中等到高分辨率（64×64+）上改进 NLL
2. **Learned variance** 让网络预测插值系数 $v$，在 $\log\beta_t$ 与 $\log\tilde\beta_t$ 间插值
3. **Hybrid loss** 隔离 $\mathcal{L}_{\text{vlb}}$ 对噪声预测的干扰
4. **Importance sampling** 按 loss 历史平方加权采样 $t$，让训练聚焦困难时间步
5. **Scaling law** 在扩散模型上同样成立——堆参数有效
6. **意外好处**：learned variance 让少步采样质量更好

---

## §9 课后任务

### 必做

1. **代码任务**：在 Project 1 基础上，添加 cosine schedule 选项
   - 实现 `cosine_beta_schedule()`
   - 在 CIFAR-10 上对比 linear vs cosine 训练曲线
   - 提交 schedule 对比图与简短分析（1 页）

2. **思考题**：
   - (a) 为什么 hybrid loss 中要把 $\lambda$ 设得很小（0.001）？如果设为 1 会怎样？
   - (b) 学方差时为什么在 log-space 插值，而不是直接在线性空间预测 $\sigma^2$？

3. **Reading note**：阅读 Improved DDPM 论文 §3-§4，按模板提交笔记。重点理解 learned variance 的动机与实现。

### 选做

4. **进阶代码**：实现 learned variance + hybrid loss，在 CIFAR-10 上训练
   - 修改 U-Net 输出 2C 通道
   - 实现 vlb loss
   - 对比 fixed vs learned variance 的 100-step FID
   - 这是论文核心实验的复现

5. **Importance sampling 实验**：实现 `LossAwareSampler`，对比均匀采样的训练收敛速度。

---

## §10 推荐进一步阅读

| 资源 | 重点 |
|------|------|
| Improved DDPM 论文全文 | 必读 |
| 推导手稿 `derive_03_ddpm_loss.pdf` | hybrid loss 的 vlb 项推导 |
| Stable Diffusion 1.5 的 schedule 设置 | 工业级 schedule 选择 |
| EDM 论文（Karras 2022）的 schedule 章节 | 进阶视角：从 SNR 重新设计 schedule |

---

## §11 常见问题

**Q: 我在 CIFAR-10 上跑 cosine 反而 FID 变差了？**

A: 这是已知现象。Cosine 设计针对 64×64+ 分辨率，在 32×32 的 CIFAR-10 上改进很小，方差还可能让 FID 略差 1-2 点。不要硬要用 cosine——基于实测选择。

**Q: Learned variance 工程上稳定吗？**

A: 大部分时候稳定，但偶尔会 collapse（$v$ 变成全 0 或全 1）。解决方法：
- $\lambda$ 不要太大
- 训练初期可以先 freeze variance 通道（只学噪声预测），后期再 unfreeze

**Q: Stable Diffusion 用了 learned variance 吗？**

A: 没有。SD 1.x/2.x 仍然用 fixed variance。Learned variance 在产业实践中并不普及，原因之一是 SD 的目标主要是 FID 和文本对齐，对 NLL 不敏感。

**Q: 论文里说 cosine 是为了"让低频内容在加噪后期保留"，怎么理解？**

A: cosine schedule 让中间 $t$ 区域 $\bar\alpha_t$ 衰减更慢。这意味着模型在中等噪声水平上看到更多训练样本，其中粗略结构（低频）刚好处于"接近被噪声覆盖但仍可见"的状态——这是学习低频结构的关键区间。

---

> **下一讲预告**：L06 我们将进入 **Score SDE**（Song et al. 2021），看到 DDPM 与 score matching 如何在连续时间 SDE 视角下统一。这是整个课程的"数学高峰"，也是后续 flow matching、consistency model 的理论基础。
