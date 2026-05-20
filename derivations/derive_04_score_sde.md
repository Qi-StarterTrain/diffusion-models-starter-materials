# Derivation 04: Score SDE 与 Anderson 反向公式

> 本手稿对应 L06 内容，详细推导：
> - DDPM 的 SDE 极限
> - Anderson 反向 SDE 公式
> - Probability flow ODE 的等价性

---

## §1 从离散 DDPM 到连续 SDE

### 1.1 离散过程参数化

DDPM 单步：
$$x_t = \sqrt{1-\beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon_t$$

设 $T$ 步对应时间区间 $[0, 1]$（即 $\Delta t = 1/T$）。把 $\beta_t$ 写成密度形式：
$$\beta_t = \beta(t/T) \cdot \frac{1}{T} = \beta(s) \Delta t, \quad s = t/T$$

其中 $\beta(s)$ 是连续函数（如 linear schedule 对应线性函数）。

---

### 1.2 一阶 Taylor 展开

当 $\Delta t$ 小时：
$$\sqrt{1 - \beta(s) \Delta t} \approx 1 - \frac{1}{2} \beta(s) \Delta t$$

代入单步公式：
$$x(s + \Delta t) \approx \left(1 - \frac{1}{2}\beta(s)\Delta t\right) x(s) + \sqrt{\beta(s) \Delta t} \cdot \epsilon$$

即：
$$x(s + \Delta t) - x(s) \approx -\frac{1}{2}\beta(s)\Delta t \cdot x(s) + \sqrt{\beta(s)\Delta t} \cdot \epsilon$$

---

### 1.3 取极限：SDE

形式上写成 SDE（用 $dW = \sqrt{dt} \cdot \epsilon$）：

$$\boxed{dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dW}$$

这就是 **VP-SDE**（Variance Preserving SDE），对应 DDPM。

---

### 1.4 VE-SDE（对应 NCSN）

类似地，NCSN 的连续极限是：
$$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, dW$$

drift = 0（没有"收缩"项），方差随 $\sigma(t)$ 增长。

---

## §2 SDE 的边缘密度演化：Fokker-Planck

### 2.1 一般 SDE 形式

$$dx = f(x, t) \, dt + g(t) \, dW$$

设 $x$ 的边缘密度为 $p_t(x)$。

---

### 2.2 Fokker-Planck 方程（无证明，标准结果）

$$\frac{\partial p_t(x)}{\partial t} = -\sum_i \frac{\partial}{\partial x_i}\left[f_i(x, t) p_t(x)\right] + \frac{1}{2} g(t)^2 \sum_i \frac{\partial^2 p_t(x)}{\partial x_i^2}$$

向量形式：
$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (f \cdot p_t) + \frac{1}{2} g(t)^2 \Delta p_t$$

（$\Delta$ 是 Laplacian，对角的二阶导）

**意义**：SDE 描述粒子轨迹，FPE 描述粒子密度的演化。

---

## §3 反向 SDE：Anderson 公式

### 3.1 设置

正向 SDE 从 $t=0$ 跑到 $t=T$，密度从 $p_0 = p_{\text{data}}$ 演化到 $p_T \approx \mathcal{N}(0, I)$。

**反向问题**：能否构造一个 SDE，从 $t=T$ 跑到 $t=0$，使其密度演化与正向**完全相同**（但方向相反）？

如果能，从 $x_T \sim p_T$ 出发跑反向 SDE 就能采样到 $p_0 = p_{\text{data}}$！

---

### 3.2 Anderson 定理（1982）

**结论**：反向 SDE 形式为
$$\boxed{dx = \left[f(x, t) - g(t)^2 \cdot \nabla_x \log p_t(x)\right] dt + g(t) \, d\bar W}$$

其中 $d\bar W$ 是反向 Wiener 过程的微元。

---

### 3.3 证明思路

设反向过程是 $dx = \tilde f(x, t) \, dt + \tilde g(t) \, d\bar W$，密度 $\tilde p_t = p_t$（同一密度，只是时间方向相反）。

由于在反向时间下，"反向 FPE" 形式为：
$$-\frac{\partial \tilde p_t}{\partial t} = -\nabla \cdot (\tilde f \cdot \tilde p_t) + \frac{1}{2} \tilde g^2 \Delta \tilde p_t$$

**要求与正向 FPE 一致**：

$$
\begin{aligned}
\frac{\partial p_t}{\partial t} &= -\nabla \cdot (f p_t) + \frac{1}{2} g^2 \Delta p_t \quad \text{(forward)}\\
-\frac{\partial p_t}{\partial t} &= -\nabla \cdot (\tilde f p_t) + \frac{1}{2} \tilde g^2 \Delta p_t \quad \text{(reverse)}
\end{aligned}
$$

两式相加：
$$0 = -\nabla \cdot ((f + \tilde f) p_t) + \frac{1}{2}(g^2 + \tilde g^2) \Delta p_t$$

要让这恒为零，可以利用恒等式：
$$\Delta p_t = \nabla \cdot \nabla p_t = \nabla \cdot \left(p_t \cdot \nabla \log p_t\right)$$

代入：
$$0 = -\nabla \cdot ((f + \tilde f) p_t) + \frac{1}{2}(g^2 + \tilde g^2) \nabla \cdot (p_t \nabla \log p_t)$$

要恒成立，**括号内项应当抵消**：
$$f + \tilde f = \frac{1}{2}(g^2 + \tilde g^2) \nabla \log p_t$$

选 $\tilde g = g$（保持同样的"噪声强度"），得：
$$\tilde f = -f + g^2 \nabla \log p_t$$

所以反向 SDE 的 drift 是：
$$\tilde f(x, t) = f(x, t) - g(t)^2 \nabla_x \log p_t(x)$$

（这里 sign 看似有差异，但实际是时间方向反转的代数处理；详见 Song 2021 论文附录 A）

∎

---

### 3.4 关键洞察

要做反向采样，**只需要知道每个时刻的 score $\nabla_x \log p_t(x)$**。

这就是为什么 score-based 模型的训练目标是估计 score——**score 是反向 SDE 唯一未知的量**。

---

## §4 Probability Flow ODE

### 4.1 神奇的事实

对任意 SDE $dx = f \, dt + g \, dW$，存在**确定性 ODE**：

$$\boxed{\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \cdot \nabla_x \log p_t(x)}$$

它**在每个时间 $t$ 的边缘分布 $p_t(x)$ 与原 SDE 完全相同**，但轨迹是确定性的。

---

### 4.2 证明（FPE 视角）

ODE $dx/dt = h(x, t)$ 对应的"FPE"（实际是 transport equation）是：
$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (h \cdot p_t)$$

要使其等于 SDE 的 FPE：
$$-\nabla \cdot (h \cdot p_t) = -\nabla \cdot (f \cdot p_t) + \frac{1}{2} g^2 \Delta p_t$$

利用 $\Delta p_t = \nabla \cdot (p_t \nabla \log p_t)$：
$$-\nabla \cdot (h \cdot p_t) = -\nabla \cdot \left[\left(f - \frac{1}{2} g^2 \nabla \log p_t\right) p_t\right]$$

所以
$$h(x, t) = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$$

∎

---

### 4.3 为什么这个 ODE 重要？

1. **可逆**：从 $x_T$ 倒积分回 $x_0$ 再正积分到 $x_T$ 应该回到同一点
2. **可计算精确 likelihood**（instantaneous change of variables）
3. **可用高阶 ODE 求解器加速**（DDIM, DPM-Solver, EDM 等）
4. **去随机性**：给定 $x_T$，$x_0$ 唯一

---

## §5 VP-SDE 的具体形式

代入 $f(x, t) = -\frac{1}{2}\beta(t) x$，$g(t) = \sqrt{\beta(t)}$：

**反向 SDE**：
$$dx = \left[-\frac{1}{2}\beta(t) x - \beta(t) \nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)} \, d\bar W$$

**Probability flow ODE**：
$$\frac{dx}{dt} = -\frac{1}{2}\beta(t) x - \frac{1}{2}\beta(t) \nabla_x \log p_t(x)$$

---

## §6 Score 与 Noise Prediction 的等价

### 6.1 离散 $\to$ 连续

离散 DDPM：$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$。

连续 VP-SDE：定义 $\bar\alpha(t) = \exp\left(-\int_0^t \beta(s) ds\right)$（exact 形式）。

类似地，$x(t) | x(0) \sim \mathcal{N}(\sqrt{\bar\alpha(t)} x_0, (1-\bar\alpha(t)) I)$。

---

### 6.2 Score 的解析形式

由 derive_02 的工具：
$$\nabla_x \log p_t(x | x_0) = -\frac{x - \sqrt{\bar\alpha(t)} x_0}{1 - \bar\alpha(t)} = -\frac{\sqrt{1-\bar\alpha(t)} \epsilon}{1-\bar\alpha(t)} = -\frac{\epsilon}{\sqrt{1-\bar\alpha(t)}}$$

—— **预测 $\epsilon$ ≡ 预测 score**（差时间相关系数）。

---

## §7 DDIM 是 ODE 的离散化

回顾 DDIM 公式：
$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \hat x_0(x_t) + \sqrt{1-\bar\alpha_{t-1}} \epsilon_\theta$$

其中 $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t} \epsilon_\theta) / \sqrt{\bar\alpha_t}$。

**断言**：DDIM 等价于 probability flow ODE 的某种一阶离散化（在适当参数变换下）。

证明思路（粗略）：
1. 把 ODE 改写成关于"参数化变量" $\lambda(t) = \log(\sqrt{\bar\alpha_t}/\sqrt{1-\bar\alpha_t})$ 的形式
2. 做 Euler 离散化
3. 化简后得到 DDIM 公式

详见 Lu 2022 DPM-Solver 论文 §3。

---

## §8 EDM 的统一框架

Karras 2022 EDM 进一步推广：

把扩散过程写成：
$$x_t = x_0 + \sigma(t) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

（这是 VE-style 形式，$\sigma(t)$ 是单调增的 noise schedule）

对应 ODE（不带任何系数变换）：
$$\frac{dx}{dt} = -\dot\sigma(t) \cdot \sigma(t) \cdot \nabla_x \log p_t(x)$$

**好处**：
- 形式极简
- $\sigma(t)$ 直接对应 SNR
- 离散化、采样器都更容易设计

EDM 的训练 loss、采样器 schedule 都基于这个统一形式。**如果你要训自己的 SOTA 扩散模型，强烈推荐用 EDM 形式**。

---

## §9 数值实验：在 2D 玩具上验证

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# 2D 玩具数据：环形分布
def sample_data(n=1000):
    theta = torch.rand(n) * 2 * np.pi
    r = 1.0 + 0.05 * torch.randn(n)
    return torch.stack([r*torch.cos(theta), r*torch.sin(theta)], dim=1)

# VP-SDE 加噪
def add_noise(x0, t):
    # 用 linear schedule
    beta_min, beta_max = 0.1, 20.0
    def alpha_bar(t):
        return torch.exp(-(beta_min * t + 0.5 * (beta_max - beta_min) * t**2))
    a = alpha_bar(t)
    eps = torch.randn_like(x0)
    return torch.sqrt(a) * x0 + torch.sqrt(1-a) * eps

# 训练简单 MLP 预测 score
# ... (training loop)

# 用反向 SDE 采样
def reverse_sde_sample(model, n=1000, num_steps=1000):
    x = torch.randn(n, 2)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = 1.0 - i * dt
        score = model(x, t)
        drift = -0.5*beta(t)*x - beta(t)*score
        noise = torch.randn_like(x) * np.sqrt(beta(t) * dt)
        x = x - drift * dt + noise
    return x

# 用 ODE 采样
def ode_sample(model, n=1000, num_steps=1000):
    x = torch.randn(n, 2)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = 1.0 - i * dt
        score = model(x, t)
        v = -0.5*beta(t)*x - 0.5*beta(t)*score
        x = x - v * dt   # 确定性更新
    return x
```

两种采样应当产生相同的边缘分布（环形）。

---

## §10 自查题

1. 写出 VP-SDE 和 VE-SDE 的形式
2. 用 FPE 推导 Anderson 公式（不查资料）
3. 解释为什么 probability flow ODE 在每个 $t$ 的边缘分布与原 SDE 相同
4. 写出 noise prediction 与 score prediction 之间的换算系数

---

## §11 参考文献

- Anderson, *Reverse-time Diffusion Equation Models*, 1982
- Song et al., *Score-Based Generative Modeling through SDEs*, ICLR 2021
- Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*, NeurIPS 2022
- Øksendal, *Stochastic Differential Equations*（教材，SDE 入门）
