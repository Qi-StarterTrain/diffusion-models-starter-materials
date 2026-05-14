# Derivation 02: DDPM Forward Process 闭合形式完整推导

> 本手稿对应 L03 内容，详细推导 DDPM 中两个关键闭合形式：
> - $q(x_t | x_0)$
> - $q(x_{t-1} | x_t, x_0)$

---

## §1 基本设置（回顾）

### 1.1 Forward 单步定义

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \cdot I)$$

等价的重参数化形式：

$$x_t = \sqrt{1-\beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

### 1.2 符号

- $\alpha_t := 1 - \beta_t$
- $\bar\alpha_t := \prod_{s=1}^t \alpha_s$

---

## §2 关键定理一：$q(x_t | x_0)$ 闭合形式

### 2.1 命题

$$q(x_t | x_0) = \mathcal{N}\bigl(x_t; \sqrt{\bar\alpha_t} \cdot x_0, (1 - \bar\alpha_t) \cdot I\bigr)$$

等价地，**重参数化形式**：

$$x_t = \sqrt{\bar\alpha_t} \cdot x_0 + \sqrt{1-\bar\alpha_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

---

### 2.2 归纳法证明

**Base case ($t=1$)**：

由定义，
$$x_1 = \sqrt{\alpha_1} \cdot x_0 + \sqrt{1-\alpha_1} \cdot \epsilon_1, \quad \epsilon_1 \sim \mathcal{N}(0, I)$$

由 $\bar\alpha_1 = \alpha_1$，所以
$$x_1 = \sqrt{\bar\alpha_1} \cdot x_0 + \sqrt{1-\bar\alpha_1} \cdot \epsilon_1$$

命题成立。✓

---

**Inductive step**：假设对 $t-1$ 命题成立，即
$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \cdot x_0 + \sqrt{1-\bar\alpha_{t-1}} \cdot \tilde\epsilon$$
其中 $\tilde\epsilon \sim \mathcal{N}(0, I)$（独立于历史噪声）。

代入单步公式：
$$
\begin{aligned}
x_t &= \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{1-\alpha_t} \cdot \epsilon_t \\
&= \sqrt{\alpha_t} \cdot \left(\sqrt{\bar\alpha_{t-1}} x_0 + \sqrt{1-\bar\alpha_{t-1}} \tilde\epsilon\right) + \sqrt{1-\alpha_t} \cdot \epsilon_t \\
&= \sqrt{\alpha_t \bar\alpha_{t-1}} \cdot x_0 + \underbrace{\sqrt{\alpha_t(1-\bar\alpha_{t-1})} \cdot \tilde\epsilon + \sqrt{1-\alpha_t} \cdot \epsilon_t}_{=: A}
\end{aligned}
$$

注意 $\alpha_t \cdot \bar\alpha_{t-1} = \bar\alpha_t$（定义）。

---

**化简 $A$**：

$A$ 是两个**独立**高斯的线性组合：
- 第一个系数：$\sqrt{\alpha_t(1-\bar\alpha_{t-1})}$
- 第二个系数：$\sqrt{1-\alpha_t}$

由"独立高斯之和仍是高斯，方差相加"：

$$A \sim \mathcal{N}(0, \sigma^2 I)$$

其中
$$\sigma^2 = \alpha_t(1-\bar\alpha_{t-1}) + (1-\alpha_t) = \alpha_t - \alpha_t \bar\alpha_{t-1} + 1 - \alpha_t = 1 - \bar\alpha_t$$

所以可以写
$$A = \sqrt{1 - \bar\alpha_t} \cdot \epsilon$$
其中 $\epsilon \sim \mathcal{N}(0, I)$（注意：这个 $\epsilon$ 已不再是单个 $\epsilon_t$ 或 $\tilde\epsilon$，而是它们的"等效合并"）。

---

**结论**：
$$x_t = \sqrt{\bar\alpha_t} \cdot x_0 + \sqrt{1-\bar\alpha_t} \cdot \epsilon$$

由 $\epsilon \sim \mathcal{N}(0, I)$，命题对 $t$ 成立。

由数学归纳法，命题对所有 $t \geq 1$ 成立。∎

---

### 2.3 直观理解

$x_t$ 是 $x_0$ 与噪声的线性组合：
- **信号部分**：$\sqrt{\bar\alpha_t} \cdot x_0$，随 $t$ 单调减小
- **噪声部分**：$\sqrt{1-\bar\alpha_t} \cdot \epsilon$，随 $t$ 单调增大
- 两部分方差之和始终为 1（VP 性质）

---

## §3 关键定理二：$q(x_{t-1} | x_t, x_0)$ 闭合形式

### 3.1 命题

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}\bigl(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t \cdot I\bigr)$$

其中
$$\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t} \cdot x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} \cdot x_t$$

$$\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \cdot \beta_t$$

---

### 3.2 推导（贝叶斯定理 + 高斯运算）

由贝叶斯：
$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

由马尔可夫性，$q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$。

把三个高斯密度写出来：

$$q(x_t | x_{t-1}) \propto \exp\left(-\frac{\|x_t - \sqrt{\alpha_t} x_{t-1}\|^2}{2\beta_t}\right)$$

$$q(x_{t-1} | x_0) \propto \exp\left(-\frac{\|x_{t-1} - \sqrt{\bar\alpha_{t-1}} x_0\|^2}{2(1-\bar\alpha_{t-1})}\right)$$

$$q(x_t | x_0) \propto \exp\left(-\frac{\|x_t - \sqrt{\bar\alpha_t} x_0\|^2}{2(1-\bar\alpha_t)}\right)$$

---

**指数中的项汇总**（把 $x_{t-1}$ 看作变量）：

$$\text{exponent} = -\frac{\|x_t - \sqrt{\alpha_t} x_{t-1}\|^2}{2\beta_t} - \frac{\|x_{t-1} - \sqrt{\bar\alpha_{t-1}} x_0\|^2}{2(1-\bar\alpha_{t-1})} + \text{terms not involving } x_{t-1}$$

把含 $x_{t-1}$ 的项展开：

$$
\begin{aligned}
&-\frac{\|x_t - \sqrt{\alpha_t} x_{t-1}\|^2}{2\beta_t}
= -\frac{1}{2\beta_t}\left[\alpha_t \|x_{t-1}\|^2 - 2\sqrt{\alpha_t} x_t^\top x_{t-1} + \|x_t\|^2\right] \\
&-\frac{\|x_{t-1} - \sqrt{\bar\alpha_{t-1}} x_0\|^2}{2(1-\bar\alpha_{t-1})}
= -\frac{1}{2(1-\bar\alpha_{t-1})}\left[\|x_{t-1}\|^2 - 2\sqrt{\bar\alpha_{t-1}} x_0^\top x_{t-1} + \|x_0\|^2 \bar\alpha_{t-1}\right]
\end{aligned}
$$

合并含 $\|x_{t-1}\|^2$ 的项：

$$
-\frac{1}{2}\left[\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}}\right] \|x_{t-1}\|^2
$$

化简系数：
$$
\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}}
= \frac{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}{\beta_t(1-\bar\alpha_{t-1})}
$$

注意到 $\alpha_t(1-\bar\alpha_{t-1}) + \beta_t = \alpha_t - \alpha_t \bar\alpha_{t-1} + 1 - \alpha_t = 1 - \bar\alpha_t$。

所以系数为：
$$
\frac{1 - \bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}
$$

—— 这是逆方差 $1/\tilde\beta_t$！得到：

$$\tilde\beta_t = \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

✓ 与命题一致。

---

**合并含 $x_{t-1}$ 的线性项**：

$$
\frac{\sqrt{\alpha_t}}{\beta_t} x_t^\top x_{t-1} + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}} x_0^\top x_{t-1}
$$

**配方法**：上式整体形式是 $-\frac{1}{2\tilde\beta_t} \|x_{t-1} - \tilde\mu_t\|^2 + \text{const}$，所以
$$\frac{\tilde\mu_t}{\tilde\beta_t} = \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}} x_0$$

即
$$\tilde\mu_t = \tilde\beta_t \cdot \left[\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}} x_0\right]$$

代入 $\tilde\beta_t = \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$：

$$
\begin{aligned}
\tilde\mu_t &= \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} \cdot \frac{\sqrt{\alpha_t}}{\beta_t} \cdot x_t + \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} \cdot \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}} \cdot x_0 \\
&= \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} \cdot x_t + \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1-\bar\alpha_t} \cdot x_0
\end{aligned}
$$

✓ 与命题一致。

---

### 3.3 改写：用 $\epsilon$ 而非 $x_0$ 参数化

由定理一：$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$，反解：
$$x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t - \sqrt{1-\bar\alpha_t} \epsilon\right)$$

代入 $\tilde\mu_t$：

$$
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1-\bar\alpha_t} \cdot \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t - \sqrt{1-\bar\alpha_t} \epsilon\right) + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t
$$

注意 $\sqrt{\bar\alpha_{t-1}} / \sqrt{\bar\alpha_t} = 1/\sqrt{\alpha_t}$，化简后：

$$\boxed{\tilde\mu_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \cdot \epsilon\right)}$$

**化简过程**（中间步）：

$x_t$ 的系数为 $\frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}} + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$

$= \frac{1}{(1-\bar\alpha_t)\sqrt{\alpha_t}}[\beta_t + \alpha_t(1-\bar\alpha_{t-1})]$  

$= \frac{1-\bar\alpha_t}{(1-\bar\alpha_t)\sqrt{\alpha_t}}$（用 $\beta_t + \alpha_t(1-\bar\alpha_{t-1}) = 1-\bar\alpha_t$）

$= \frac{1}{\sqrt{\alpha_t}}$ ✓

$\epsilon$ 的系数为 $-\frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}} \cdot \sqrt{1-\bar\alpha_t}$

$= -\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}$ ✓

---

### 3.4 启示

均值的形式：$\tilde\mu_t$ 是 $x_t$ 减去一个跟 $\epsilon$ 成比例的项。

> **这指引了 DDPM 的网络设计**：让 $\epsilon_\theta(x_t, t)$ 预测 $\epsilon$，则反向均值 $\mu_\theta$ 就有显式形式，简化训练目标。

下一份推导手稿 (derive_03) 将完成 ELBO → MSE 的化简。

---

## §4 数值验证（强烈建议自己跑一遍）

```python
import torch

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alphas_cumprod = alphas.cumprod(0)

# 验证定理一：q(x_t | x_0) 闭合形式
def verify_q_xt(x0, t, num_trials=10000):
    direct_samples = []
    for _ in range(num_trials):
        x = x0.clone()
        for s in range(t + 1):
            eps = torch.randn_like(x)
            x = torch.sqrt(alphas[s]) * x + torch.sqrt(betas[s]) * eps
        direct_samples.append(x)
    direct_samples = torch.stack(direct_samples)

    # vs 闭合形式
    closed_form = []
    for _ in range(num_trials):
        eps = torch.randn_like(x0)
        x = torch.sqrt(alphas_cumprod[t]) * x0 + torch.sqrt(1 - alphas_cumprod[t]) * eps
        closed_form.append(x)
    closed_form = torch.stack(closed_form)

    print(f"t = {t}")
    print(f"  direct mean: {direct_samples.mean():.4f}, var: {direct_samples.var():.4f}")
    print(f"  closed mean: {closed_form.mean():.4f}, var: {closed_form.var():.4f}")
    # 应当几乎一致

x0 = torch.zeros(1)
for t in [10, 100, 500, 999]:
    verify_q_xt(x0, t)
```

---

## §5 自查题

1. 不查资料推出 $q(x_t | x_0)$ 闭合形式
2. 不查资料推出 $\tilde\beta_t$ 的表达式
3. 解释为什么 $\tilde\mu_t$ 的形式"指引"了网络预测 $\epsilon$ 而非 $x_{t-1}$

---

## §6 参考文献

- Ho et al., *DDPM*, NeurIPS 2020, §2-§3
- Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, ICML 2015
