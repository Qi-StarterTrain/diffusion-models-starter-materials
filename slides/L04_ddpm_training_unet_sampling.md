# Lecture 04｜DDPM 下：训练目标、U-Net 架构、采样算法

> **本讲目标**
> - 完整推导 DDPM 的训练目标，从 ELBO 化简到极简的 MSE
> - 理解 U-Net 在 DDPM 中的具体设计（timestep embedding、attention、ResBlock）
> - 写出完整的训练与采样伪代码
> - 理解几个常见的工程技巧（EMA、混合精度）背后的原理

> **前置要求**：必须已完成 L03 的手推作业。本讲推导高度依赖 L03 中的两个闭合形式。

---

## §1 整体回顾与本讲路线图

L03 我们建立了：
- 前向过程：$q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, \beta_t I)$
- 闭合形式：$q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$
- 理想反向后验：$q(x_{t-1}|x_t, x_0) = \mathcal{N}(\tilde\mu_t, \tilde\beta_t I)$

本讲要解决：
1. **如何参数化 $p_\theta(x_{t-1}|x_t)$？**（让网络拟合反向过程）
2. **如何训练这个网络？**（ELBO → MSE）
3. **U-Net 怎么设计？**（架构细节）
4. **如何采样？**（推理算法）

---

## §2 反向过程的参数化

### 2.1 基本设计

DDPM 假设反向过程也是高斯：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}\bigl(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\bigr)$$

**两种简化**：
- DDPM 把方差固定为 $\Sigma_\theta = \sigma_t^2 I$（$\sigma_t^2 = \beta_t$ 或 $\tilde\beta_t$，性能差不多）
- 网络只预测均值 $\mu_\theta(x_t, t)$

> **改进 DDPM** 让网络也预测方差，叫 learned variance。后续讲。

---

### 2.2 完整生成过程

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} | x_t)$$

其中 $p(x_T) = \mathcal{N}(0, I)$（先验，纯噪声）。

最终生成 $x_0$ 的分布：
$$p_\theta(x_0) = \int p_\theta(x_{0:T}) \mathrm{d}x_{1:T}$$

—— 我们想最大化这个 likelihood。但和 VAE 一样，直接计算积分不现实。所以转向 ELBO。

---

## §3 训练目标推导（核心中的核心）

### 3.1 ELBO 写出

仿照 VAE，引入"变分后验"$q(x_{1:T}|x_0)$（这里就是已知的前向过程！）：

$$
\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] =: \mathcal{L}
$$

把分子分母展开：

$$
\mathcal{L} = \mathbb{E}_q\left[\log p(x_T) + \sum_{t=1}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]
$$

—— 这一步对 ELBO 公式不熟可以参考推导手稿 derive_03。

---

### 3.2 关键技巧：把 forward 改写为含 $x_0$ 的形式

直接处理上面的 sum 复杂。**核心 trick** 是把 $q(x_t|x_{t-1})$ 用 Bayes 改写：

$$q(x_t | x_{t-1}) = q(x_t | x_{t-1}, x_0) = \frac{q(x_{t-1} | x_t, x_0) \cdot q(x_t | x_0)}{q(x_{t-1} | x_0)}$$

代入 ELBO 后，经过一些代数操作（详见 derive_03），可以化简成：

$$
\mathcal{L} = \mathbb{E}_q\Bigl[\underbrace{D_{\mathrm{KL}}(q(x_T|x_0) \| p(x_T))}_{L_T}\Bigr]
+ \sum_{t=2}^T \mathbb{E}_q\Bigl[\underbrace{D_{\mathrm{KL}}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}}\Bigr]
- \mathbb{E}_q\Bigl[\underbrace{\log p_\theta(x_0|x_1)}_{L_0}\Bigr]
$$

每项的含义：

| 项 | 含义 |
|----|------|
| $L_T$ | 终态对齐项（不含 $\theta$，可忽略） |
| $L_{t-1}$ for $t=2,\dots,T$ | 让 $p_\theta(x_{t-1}\|x_t)$ 逼近 $q(x_{t-1}\|x_t, x_0)$ |
| $L_0$ | 重构项（最后一步） |

**关键观察**：$L_{t-1}$ 是**两个高斯之间的 KL**，可以闭合计算。这就是 DDPM 训练目标极简的根源。

---

### 3.3 高斯 KL 化简

回顾 §L03，
- $q(x_{t-1} | x_t, x_0) = \mathcal{N}(\tilde\mu_t(x_t, x_0), \tilde\beta_t I)$
- $p_\theta(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 I)$

两个等方差高斯的 KL：
$$D_{\mathrm{KL}}(\mathcal{N}(\mu_1, \sigma^2 I) \| \mathcal{N}(\mu_2, \sigma^2 I)) = \frac{\| \mu_1 - \mu_2 \|^2}{2\sigma^2}$$

故：
$$L_{t-1} = \mathbb{E}_q \left[ \frac{\| \tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t) \|^2}{2\sigma_t^2} \right]$$

—— **训练目标变成了 MSE**！

---

### 3.4 进一步化简：从均值预测到噪声预测

代入 L03 推出的 $\tilde\mu_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon \right)$，

并把 $\mu_\theta$ **设计为相同的形式**：
$$\mu_\theta(x_t, t) := \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t, t) \right)$$

—— 让网络预测 $\epsilon$ 而不是均值。

代入 KL：
$$L_{t-1} = \mathbb{E}_{x_0, \epsilon} \left[ \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1-\bar\alpha_t)} \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

—— **核心目标**：噪声的 MSE，加一个时间相关的权重。

---

### 3.5 DDPM 的最终训练目标（**最重要的公式**）

DDPM 论文进一步发现：**去掉权重，在所有 $t$ 上等权重训练，效果反而更好**。

最终极简训练目标：

$$\boxed{\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t \sim U[1,T], \, x_0, \, \epsilon \sim \mathcal{N}(0,I)} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon, t) \|^2 \right]}$$

**这就是 DDPM 的全部训练算法**。看似简单，背后是 ELBO + 高斯 KL + 重参数化的全套机器。

---

### 3.6 为什么去掉权重效果更好？

权重 $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar\alpha_t)}$ 在小 $t$ 时极大、大 $t$ 时较小。

这意味着：
- 加权目标过分关注**已经很简单的小 $t$ 任务**（去除一点点噪声）
- 忽略了**真正困难的大 $t$ 任务**（从接近纯噪声中恢复结构）

去掉权重相当于给所有时间步**等量训练资源**，反而让大 $t$ 区域学得更好。

---

## §4 训练算法（Algorithm 1）

```
DDPM Training:
─────────────────────────────────────
repeat:
    1. x_0 ~ p_data(x_0)              # 从数据集采样一张图
    2. t ~ Uniform({1, ..., T})        # 随机时间步
    3. ε ~ N(0, I)                     # 随机噪声
    4. x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε   # 一步加噪到 t
    5. loss = ‖ε - ε_θ(x_t, t)‖²       # 预测噪声的 MSE
    6. 反向传播 + 优化器更新
until converged
─────────────────────────────────────
```

**对应 PyTorch 代码框架**：

```python
def p_losses(model, x0, t, betas):
    epsilon = torch.randn_like(x0)
    sqrt_alpha_bar_t = sqrt_alpha_bar[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

    x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * epsilon
    pred = model(x_t, t)
    return F.mse_loss(pred, epsilon)
```

—— 整个训练逻辑就这么简洁！

---

## §5 U-Net 架构详解

### 5.1 为什么用 U-Net？

DDPM 的网络要做"去噪"——给定 $x_t$ 和 $t$，输出 $\epsilon$（同 shape 的张量）。

U-Net 天然适合此任务：
- 输入输出同形状
- 编码器-解码器结构能融合多尺度信息
- skip connection 保留细节信息（去噪需要这些）

---

### 5.2 DDPM U-Net 的关键模块

```
Input x_t (B, 3, 32, 32)
  │
  ├─ Conv (3 → 128 channels)
  │
  ├─ Down stage 1 (128, 32x32)
  │     ResBlock × 2
  │     ─── skip connection ───┐
  │     Downsample (32→16)     │
  │                             │
  ├─ Down stage 2 (256, 16x16)  │
  │     ResBlock × 2            │
  │     Attention (16x16)        │
  │     ─── skip connection ──┐  │
  │     Downsample (16→8)     │  │
  │                            │  │
  ├─ Down stage 3 (512, 8x8)   │  │
  │     ...                     │  │
  │                              │  │
  ├─ Mid (Bottleneck)            │  │
  │     ResBlock + Attention      │  │
  │                                │  │
  ├─ Up stage 3 (concat skip) ◀───┘  │
  │     ResBlock × 2                  │
  │     Upsample (8→16)               │
  │                                   │
  ├─ Up stage 2 (concat skip) ◀──────┘
  │     ...
  │
  ├─ Up stage 1
  │     ...
  │
  └─ Conv → Output ε (B, 3, 32, 32)

Time embedding t (B,) → MLP → broadcast 到每个 ResBlock
```

---

### 5.3 Timestep Embedding

时间步 $t$ 是离散整数，需要变成连续向量后注入网络。

**Sinusoidal embedding**（Transformer 同款）：

$$\text{PE}(t)_{2k} = \sin\left(\frac{t}{10000^{2k/d}}\right), \quad \text{PE}(t)_{2k+1} = \cos\left(\frac{t}{10000^{2k/d}}\right)$$

**代码**：
```python
def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    return torch.cat([args.cos(), args.sin()], dim=-1)
```

之后通过 MLP 映射到目标维度，**广播加到每个 ResBlock 的 feature**。

---

### 5.4 ResBlock with Time Embedding

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = GroupNorm(in_ch)
        self.conv1 = Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = Linear(time_dim, out_ch)

        self.norm2 = GroupNorm(out_ch)
        self.conv2 = Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else Identity()

    def forward(self, x, t_emb):
        h = self.conv1(silu(self.norm1(x)))
        h = h + self.time_mlp(silu(t_emb)).view(-1, h.shape[1], 1, 1)
        h = self.conv2(silu(self.norm2(h)))
        return h + self.skip(x)
```

**关键点**：
- 时间 embedding 通过广播加法注入（不是 concat）
- 用 GroupNorm 而非 BatchNorm（小 batch 训练更稳）
- 激活用 SiLU（比 ReLU 更平滑，扩散模型中默认）

---

### 5.5 Self-Attention 的位置

Attention 计算量随分辨率平方增长，所以：
- 高分辨率（如 32x32, 16x16）：仅在某些层加 attention
- 低分辨率（如 8x8）：可以多加 attention 用来建模长程依赖

> **典型配置**（DDPM CIFAR-10）：在 16x16 分辨率层添加 self-attention。

---

## §6 采样算法（Algorithm 2）

```
DDPM Sampling:
─────────────────────────────────────
1. x_T ~ N(0, I)                       # 从纯噪声开始
2. for t = T, T-1, ..., 1:
   3.   z ~ N(0, I) if t > 1 else 0    # t=1 时无随机性
   4.   pred_ε = ε_θ(x_t, t)            # 预测噪声
   5.   μ = (1/√α_t) * (x_t - β_t/√(1-ᾱ_t) * pred_ε)  # 反向均值
   6.   x_{t-1} = μ + σ_t * z           # 加随机扰动
7. return x_0
─────────────────────────────────────
```

**对应代码**：
```python
@torch.no_grad()
def p_sample(model, x_t, t):
    pred_eps = model(x_t, t)
    mu = (1.0 / sqrt_alpha[t]) * (
        x_t - betas[t] / sqrt_one_minus_alpha_bar[t] * pred_eps
    )
    if t == 0:
        return mu  # 最后一步无随机性
    z = torch.randn_like(x_t)
    sigma = sqrt(betas[t])  # 也可用 ~β_t
    return mu + sigma * z

@torch.no_grad()
def sample_loop(model, shape):
    x = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        x = p_sample(model, x, t)
    return x
```

---

## §7 工程技巧（实战必备）

### 7.1 EMA（指数移动平均）

DDPM 论文发现：用 EMA 参数采样比直接用最后 checkpoint 效果显著更好。

**为什么？**
- 训练过程参数有噪声波动
- EMA 平滑了这些波动，给出"训练全过程的代表"
- 类似 stochastic averaging 的效果

**代码**：
```python
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply_to(self, model):
        model.load_state_dict(self.shadow)
```

**典型设置**：decay=0.9999 或 0.999；至少更新 10K 步后 EMA 才稳定。

---

### 7.2 混合精度（fp16）

显存需求大概减半，速度提升 30–50%。

**代码模板**：
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for batch in loader:
    optimizer.zero_grad()
    with autocast():
        loss = p_losses(model, batch, t)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**踩坑点**：
- GroupNorm 在 fp16 下可能 NaN，可强制 fp32 计算
- bfloat16（A100/H100）比 fp16 更稳，无需 GradScaler

---

### 7.3 梯度裁剪

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

防止偶发梯度爆炸，是 DDPM 的标准实践。

---

### 7.4 学习率与 warmup

DDPM 默认 Adam(2e-4)，前几千步线性 warmup。

```python
def lr_lambda(step):
    return min(step / warmup_steps, 1.0)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

## §8 完整训练代码骨架（Project 1 起点）

```python
def train(model, loader, optimizer, schedule, num_steps):
    ema = EMA(model)
    scaler = GradScaler()

    for step in range(num_steps):
        x0 = next(loader_iter).to(device)
        t = torch.randint(0, T, (x0.shape[0],), device=device)

        optimizer.zero_grad()
        with autocast():
            loss = p_losses(model, x0, t, schedule)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        ema.update(model)

        if step % log_freq == 0:
            wandb.log({'loss': loss.item()}, step=step)

        if step % sample_freq == 0:
            ema.apply_to(eval_model)
            samples = sample_loop(eval_model, (16, 3, 32, 32))
            wandb.log({'samples': wandb.Image(samples)}, step=step)
```

---

## §9 本讲核心要点回顾

1. ELBO 化简到 KL of 高斯，再化简到 MSE，最终得到极简噪声预测目标
2. 等权重的 simplified loss 比 ELBO 加权效果更好——优先训练困难时间步
3. U-Net + 时间 embedding + 部分层 attention 是 DDPM 标准架构
4. 训练算法 8 行、采样算法 8 行——但每一行背后都有数学
5. EMA、fp16、gradient clipping 是工程必备

---

## §10 课后任务（**Project 1 启动**）

### 必做

1. **手推训练目标**：从 ELBO 出发，完整推导到 simplified MSE loss。可参考 derive_03。

2. **代码任务**（Project 1 起步）：
   按照 `starter_code/project1_ddpm/README.md` 完成基础档：
   - 实现 `q_sample`、`p_losses`、`p_sample_loop`
   - 实现 sinusoidal time embedding 与 ResBlock
   - 在 MNIST 上训练 50 epoch，生成可辨认数字

3. **思考题**：
   (a) 如果让网络预测 $x_0$ 而非 $\epsilon$，训练目标会是什么？两者数学上等价吗？
   (b) 为什么 $t=1$ 时采样不加随机性？（提示：思考 $L_0$ 项的本质）

### 选做

4. **进阶档**：在 CIFAR-10 上训练 200 epoch，生成 5000 样本计算 FID（目标 ≤ 15）。

5. **挑战档**：实现 cosine schedule 与 linear schedule 的对比实验，提交 8 页技术报告分析差异。

---

## §11 推荐进一步阅读

| 资源 | 重点 |
|------|------|
| DDPM 论文 §3-§4 | 训练目标推导 |
| Improved DDPM §3 | 详细工程改进 |
| `lucidrains/denoising-diffusion-pytorch` | 极简参考实现 |
| HuggingFace Diffusers `DDPMScheduler` | 工业级实现 |
| 推导手稿 `derive_03_ddpm_loss.pdf` | 完整 ELBO 推导 |

---

## §12 常见问题

**Q: 为什么训练时随机采样 $t$，而不是按顺序遍历？**

A: 这是单样本 MC 估计期望的标准做法。$\mathcal{L}$ 本身是对所有 $t$ 的求和（×期望），随机采样让每个 batch 是无偏估计。如果按顺序，会让训练严重不平衡。

**Q: $\sigma_t^2$ 用 $\beta_t$ 还是 $\tilde\beta_t$？**

A: 实践差不多。$\beta_t$（"upper bound"）：噪声更大，可能采样多样；$\tilde\beta_t$（"lower bound"）：更接近理想后验。两者 FID 差异在 0.5 以内。Improved DDPM 干脆让网络学习方差。

**Q: 为什么 DDPM 训练这么稳定？**

A: 三个原因：
1. 训练目标就是 MSE 回归，本质上和监督学习一样稳定
2. 加噪过程提供了很强的隐式正则（每个 batch 都看到不同噪声水平）
3. 没有 GAN 那样的博弈，不存在训练失衡

**Q: 我看到代码里 `t` 是从 0 开始的，但论文公式 $t \in \{1, \dots, T\}$。如何对应？**

A: 代码 0-indexed 是惯例，对应论文的 $t-1$。具体看代码细节。这是新手常见混淆点。

**Q: 训练时 loss 一直降不下来怎么办？**

A: 常见问题排查清单：
- ❓ 数据归一化了吗？DDPM 期望 $x_0 \in [-1, 1]$
- ❓ time embedding 真的注入到每个 ResBlock 了吗？
- ❓ schedule 张量用 `register_buffer` 注册了吗？（否则 `.to(device)` 不跟着移动）
- ❓ batch 内 $t$ 的随机性正确吗？
- ❓ 损失曲线是否进入了 0.02 附近的"地板"？这通常是正常的（MSE noise prediction 的典型量级）

---

> **下一讲预告**：L05 我们将进入 *Improved DDPM*，学习 cosine schedule、learned variance、importance sampling 等具体改进。继续往后是 Score SDE、DDIM、CFG 等。
>
> **本讲学习达成检查**：
> - [ ] 能闭眼写出 simplified DDPM loss 公式
> - [ ] 能完整推导 ELBO → MSE 的化简
> - [ ] 写出训练算法和采样算法的 PyTorch 代码骨架
> - [ ] 知道 EMA / fp16 / grad clip 的作用
> - [ ] 已开始 Project 1 的实现
