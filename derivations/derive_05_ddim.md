# Derivation 05: DDIM 完整推导

> 本手稿对应 L07 内容，详细推导 DDIM 的非马尔可夫前向过程与确定性反向公式。

---

## §1 核心洞察回顾

DDPM 的训练目标只依赖**边缘分布** $q(x_t | x_0)$：
$$\mathcal{L}_{\text{simple}} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\|^2\right]$$

不依赖完整马尔可夫链 $q(x_{1:T}|x_0)$。

**DDIM 思路**：构造一类不同的前向过程，**保持每个 $q(x_t | x_0)$ 不变**，但反向过程可以更高效（甚至确定性）。

---

## §2 设计非马尔可夫前向过程

### 2.1 目标

构造分布族 $\{q_\sigma\}$，参数化为 $\sigma = (\sigma_1, \dots, \sigma_T)$，满足：

1. **边缘保持**：$q_\sigma(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$（与 DDPM 同）
2. **后验形式**：$q_\sigma(x_{t-1} | x_t, x_0)$ 是高斯，方差为 $\sigma_t^2 I$

---

### 2.2 后验设计

要求 $q_\sigma(x_{t-1} | x_t, x_0)$ 满足边际一致性：
$$\int q_\sigma(x_{t-1} | x_t, x_0) \cdot q_\sigma(x_t | x_0) \, dx_t = q_\sigma(x_{t-1} | x_0)$$

由于左边是两个高斯的积分（线性高斯模型），结果仍是高斯。匹配右边的均值和方差，可求出 $q_\sigma(x_{t-1} | x_t, x_0)$ 的形式。

---

### 2.3 公式

通过繁琐但机械的代数（详见 Song 2020 附录 B），得：

$$q_\sigma(x_{t-1} | x_t, x_0) = \mathcal{N}(\mu_\sigma(x_t, x_0), \sigma_t^2 I)$$

其中
$$\mu_\sigma(x_t, x_0) = \sqrt{\bar\alpha_{t-1}} \cdot x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2} \cdot \frac{x_t - \sqrt{\bar\alpha_t} x_0}{\sqrt{1-\bar\alpha_t}}$$

**等价改写**（用 $\epsilon$ 形式代入 $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\epsilon)/\sqrt{\bar\alpha_t}$）：

$$\mu_\sigma(x_t, x_0) = \sqrt{\bar\alpha_{t-1}} \cdot x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2} \cdot \epsilon$$

---

### 2.4 验证边际一致性（关键检查）

设 $x_0$ 给定。设 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$ 其中 $\epsilon \sim \mathcal{N}(0, I)$。

按上述公式：
$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2} \cdot \epsilon + \sigma_t \cdot \tilde\epsilon$$

其中 $\tilde\epsilon \sim \mathcal{N}(0, I)$（独立于 $\epsilon$）。

**计算 $x_{t-1}$ 的方差**：
$$(1-\bar\alpha_{t-1}-\sigma_t^2) \cdot \text{Var}(\epsilon) + \sigma_t^2 \cdot \text{Var}(\tilde\epsilon) = 1-\bar\alpha_{t-1}-\sigma_t^2 + \sigma_t^2 = 1-\bar\alpha_{t-1}$$

✓ 与 $q(x_{t-1}|x_0)$ 的方差一致。

**均值显然为 $\sqrt{\bar\alpha_{t-1}} x_0$**，也一致。✓

—— 任意 $\sigma_t \in [0, \sqrt{1-\bar\alpha_{t-1}}]$ 都成立。

---

## §3 DDPM 与 DDIM 是同一族的两个端点

### 3.1 $\sigma_t = \tilde\beta_t$：退化为 DDPM

代入 $\sigma_t^2 = \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$，代数化简后得到的 $\mu_\sigma$ 恰好等于 DDPM 的 $\tilde\mu_t$。

证明留作练习（用 $\beta_t = 1 - \alpha_t$，逐步代入）。

---

### 3.2 $\sigma_t = 0$：确定性反向（DDIM 默认）

当 $\sigma_t = 0$ 时，反向过程**没有随机性**：

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \cdot \hat x_0 + \sqrt{1-\bar\alpha_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

其中 $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t} \epsilon_\theta)/\sqrt{\bar\alpha_t}$ 是从噪声预测推出的 $x_0$ 估计。

**这就是 DDIM 公式**。

---

## §4 训练目标不变

**关键命题**：在任意 $\sigma$ 下，DDPM 的 simplified loss 仍然是合法目标。

证明：由 derive_03 §6，simplified loss 只涉及 $q(x_t | x_0)$ 与 $\epsilon_\theta$。由设计要求 1（边缘保持），$q_\sigma(x_t | x_0)$ 与 DDPM 完全相同，所以 loss 形式不变。

**意义**：训练好的 DDPM 网络可以**直接**用于 DDIM 采样，无需重训。

---

## §5 跳步采样

### 5.1 子序列采样

选择 $S < T$ 步子序列 $\tau_1 < \tau_2 < \dots < \tau_S$（如 $\tau_i = \lfloor i \cdot T/S \rfloor$）。

每步直接从 $x_{\tau_i}$ 跳到 $x_{\tau_{i-1}}$：

$$x_{\tau_{i-1}} = \sqrt{\bar\alpha_{\tau_{i-1}}} \cdot \hat x_0(x_{\tau_i}) + \sqrt{1-\bar\alpha_{\tau_{i-1}}} \cdot \epsilon_\theta(x_{\tau_i}, \tau_i)$$

**为什么合法**：把"前向过程"重新参数化在子序列 $\{\tau_i\}$ 上仍是合法的 DDIM 过程（公式中 $t$ 与 $t-1$ 改成相邻的 $\tau_i$ 与 $\tau_{i-1}$ 即可）。

---

### 5.2 步数与质量

$S$ 越小，采样越快但质量越差。经验值：
- $S = 1000$：与 DDPM 等价，最高质量
- $S = 50$：FID 涨 1-2，视觉不可分辨
- $S = 20$：FID 明显恶化但可用
- $S = 10$：可见劣化

> **🔑 50 步是产业实践的甜点**，被 SD 默认采用。

---

## §6 ODE 视角

### 6.1 连续极限

设 $\Delta = \log\sqrt{\bar\alpha_{t-1}/\bar\alpha_t}$ 是相邻时间步的"log-SNR 差"。当 $\Delta$ 小时，DDIM 单步是某种 ODE 的离散化。

**改写 DDIM 公式**：

$$\frac{x_{t-1}}{\sqrt{\bar\alpha_{t-1}}} = \frac{x_t}{\sqrt{\bar\alpha_t}} + \left(\sqrt{\frac{1-\bar\alpha_{t-1}}{\bar\alpha_{t-1}}} - \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\right) \cdot \epsilon_\theta$$

（推导：把 DDIM 公式同除 $\sqrt{\bar\alpha_{t-1}}$，并用 $\hat x_0$ 的定义展开。）

定义新变量：
- $\bar x_t := x_t / \sqrt{\bar\alpha_t}$（"signal-normalized"）
- $\lambda_t := \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$（"noise-to-signal ratio"）

DDIM 公式变为：
$$\bar x_{t-1} = \bar x_t + (\lambda_{t-1} - \lambda_t) \cdot \epsilon_\theta(x_t, t)$$

**这是 Euler 法对 ODE**
$$\frac{d\bar x}{d\lambda} = \epsilon_\theta(\sqrt{\bar\alpha} \cdot \bar x, \lambda)$$
**的一阶离散化**。

---

### 6.2 与 Score SDE 的连接

回顾 L06 中 VP-SDE 的 probability flow ODE：
$$\frac{dx}{dt} = -\frac{1}{2}\beta(t) x - \frac{1}{2}\beta(t) \nabla_x \log p_t(x)$$

把 $\nabla_x \log p_t = -\epsilon_\theta/\sqrt{1-\bar\alpha_t}$ 代入，再做参数化变换（$\bar x = x/\sqrt{\bar\alpha}$），可以证明等价于上面的形式。

**结论**：**DDIM 是 VP-SDE 对应的 probability flow ODE 的一阶 Euler 离散化**（在适当参数化下）。

完整代数推导见 Lu 2022 DPM-Solver 论文 §3.1。

---

## §7 DDIM Inversion

### 7.1 反向问题

给定真实图像 $x_0$，找 $x_T$ 使 DDIM 从 $x_T$ 采样能精确还原 $x_0$。

### 7.2 算法

把 DDIM 反向公式**反着用**：
$$x_t = \sqrt{\bar\alpha_t} \cdot \hat x_0(x_{t-1}) + \sqrt{1-\bar\alpha_t} \cdot \epsilon_\theta(x_{t-1}, t-1)$$

**注意**：这里用 $\epsilon_\theta(x_{t-1}, t-1)$ 而非 $\epsilon_\theta(x_t, t)$（因为我们还没有 $x_t$）——这是一阶近似，引入累积误差。

```python
def ddim_inversion(x_0, model, schedule, num_steps=50):
    x = x_0
    for t in range(num_steps):
        # 用 x_{t-1} 的网络预测推断 x_t
        eps = model(x, t)
        x_0_pred = (x - schedule.sqrt_1m_alpha_bar[t] * eps) / schedule.sqrt_alpha_bar[t]
        x = schedule.sqrt_alpha_bar[t+1] * x_0_pred + schedule.sqrt_1m_alpha_bar[t+1] * eps
    return x  # x_T
```

### 7.3 应用

- **图像编辑**：invert 真实图 → 修改 prompt → re-sample
- **风格迁移**：保留 inverted $x_T$，换 prompt
- **Null-text inversion**：更精确的 prompt-aware inversion

---

## §8 一些 DDIM 的变体

### 8.1 $\eta$-DDIM

DDIM 原论文引入参数 $\eta \in [0, 1]$ 控制 $\sigma_t$：
$$\sigma_t = \eta \cdot \sqrt{\tilde\beta_t}$$

- $\eta = 0$：DDIM（确定性）
- $\eta = 1$：DDPM-like（随机）

中间 $\eta$ 在多样性和确定性之间插值。

### 8.2 Karras schedule + Heun

Karras 2022 EDM 用 Heun's method（2 阶 Runge-Kutta）替代 Euler：
1. 用 Euler 预测一步
2. 在新位置评估梯度
3. 取平均做修正

效果：相同步数下 FID 比 DDIM 低 1-2 点。

---

## §9 数值验证

```python
# 验证 DDIM 在 σ=0 下确定性
import torch
model = load_pretrained_ddpm()
x_T = torch.randn(1, 3, 32, 32)

# 跑两次 DDIM，应当完全相同
sample1 = ddim_sample(model, x_T, steps=50)
sample2 = ddim_sample(model, x_T, steps=50)
assert torch.allclose(sample1, sample2)  # ✓ 确定性

# 验证 DDIM 与 DDPM 1000 步相近
ddim_50 = ddim_sample(model, x_T, steps=50, eta=0)
ddpm_1000 = ddpm_sample(model, x_T, steps=1000)
# FID 应当相近（50 步 vs 1000 步）

# DDIM Inversion 重构误差
x_0_original = load_test_image()
x_T_inverted = ddim_inversion(x_0_original, model)
x_0_reconstructed = ddim_sample(model, x_T_inverted, steps=50)
reconstruction_error = (x_0_original - x_0_reconstructed).abs().mean()
# 约 1-3% 像素误差
```

---

## §10 自查题

1. 推出 DDIM 后验公式 $q_\sigma(x_{t-1}|x_t,x_0)$
2. 证明 $\sigma_t = \tilde\beta_t$ 时退化为 DDPM
3. 写出 DDIM 跳步公式（任意子序列 $\tau$）
4. 解释 DDIM Inversion 的累积误差来源

---

## §11 参考文献

- Song, Meng, Ermon, *DDIM*, ICLR 2021
- Lu et al., *DPM-Solver*, NeurIPS 2022（§3 提供严格的 ODE 视角）
- Karras et al., *EDM*, NeurIPS 2022（统一框架）
- Mokady et al., *Null-text Inversion*, CVPR 2023
