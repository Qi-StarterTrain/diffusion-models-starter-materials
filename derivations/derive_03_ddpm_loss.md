# Derivation 03: DDPM 训练目标 — ELBO 到 MSE 完整化简

> 本手稿对应 L04 内容，详细推导 DDPM 从 ELBO 到 simplified MSE loss 的化简。
> 这是 DDPM 数学之美的"高峰"。

> **前置**：必须已掌握 derive_02 的两个闭合形式。

---

## §1 DDPM 作为隐变量模型

模型设定：
$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} | x_t)$$

其中：
- $p(x_T) = \mathcal{N}(0, I)$（先验，固定）
- $p_\theta(x_{t-1} | x_t)$（反向，学习）

边际：
$$p_\theta(x_0) = \int p_\theta(x_{0:T}) \, dx_{1:T}$$

---

## §2 ELBO 写出

仿照 VAE，引入变分后验。**这里 $q$ 就是已知的前向过程**！

$$q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})$$

应用 ELBO（参考 derive_01）：
$$\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T} | x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)}\right]$$

展开：
$$\mathcal{L} = \mathbb{E}_q\left[\log p(x_T) + \sum_{t=1}^T \log \frac{p_\theta(x_{t-1} | x_t)}{q(x_t | x_{t-1})}\right]$$

---

## §3 重新组织 ELBO（关键技巧）

直接处理上面的求和复杂。**关键 trick** 是把 $q(x_t | x_{t-1})$ 改写：

由马尔可夫性 $q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$（左边其实是右边）。

再由贝叶斯：
$$q(x_t | x_{t-1}, x_0) = \frac{q(x_{t-1} | x_t, x_0) \cdot q(x_t | x_0)}{q(x_{t-1} | x_0)}$$

代入 ELBO（仅 $t \geq 2$ 的项）：

$$
\begin{aligned}
\sum_{t=2}^T \log \frac{p_\theta(x_{t-1} | x_t)}{q(x_t | x_{t-1})}
&= \sum_{t=2}^T \log \frac{p_\theta(x_{t-1} | x_t) \cdot q(x_{t-1} | x_0)}{q(x_{t-1} | x_t, x_0) \cdot q(x_t | x_0)} \\
&= \sum_{t=2}^T \log \frac{p_\theta(x_{t-1} | x_t)}{q(x_{t-1} | x_t, x_0)} + \sum_{t=2}^T \log \frac{q(x_{t-1} | x_0)}{q(x_t | x_0)}
\end{aligned}
$$

**化简第二个求和（telescoping）**：

$$\sum_{t=2}^T \log \frac{q(x_{t-1} | x_0)}{q(x_t | x_0)} = \log \frac{q(x_1 | x_0)}{q(x_T | x_0)}$$

（中间项相消）

---

整体代回 ELBO：

$$
\mathcal{L} = \mathbb{E}_q\left[
\log p(x_T) + \log \frac{p_\theta(x_0 | x_1)}{q(x_1 | x_0)} + \sum_{t=2}^T \log \frac{p_\theta(x_{t-1} | x_t)}{q(x_{t-1} | x_t, x_0)} + \log \frac{q(x_1 | x_0)}{q(x_T | x_0)}
\right]
$$

化简（$\log q(x_1 | x_0)$ 与 telescoping 中相消）：

$$
\mathcal{L} = \mathbb{E}_q\left[
\log \frac{p(x_T)}{q(x_T | x_0)} + \sum_{t=2}^T \log \frac{p_\theta(x_{t-1} | x_t)}{q(x_{t-1} | x_t, x_0)} + \log p_\theta(x_0 | x_1)
\right]
$$

---

## §4 三部分分解

写成更清晰的形式：

$$
\boxed{
\mathcal{L} = \underbrace{-D_{\mathrm{KL}}(q(x_T | x_0) \| p(x_T))}_{=: -L_T}
+ \sum_{t=2}^T \underbrace{-D_{\mathrm{KL}}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))}_{=: -L_{t-1}}
+ \underbrace{\mathbb{E}_q[\log p_\theta(x_0 | x_1)]}_{=: -L_0}
}
$$

最大化 $\mathcal{L}$ 等价于最小化 $L_T + \sum_{t=2}^T L_{t-1} + L_0$。

**各项含义**：
- $L_T$：终态对齐（不含 $\theta$，可忽略）
- $L_{t-1}$ for $t = 2, \dots, T$：让 $p_\theta(x_{t-1} | x_t)$ 逼近真实后验 $q(x_{t-1} | x_t, x_0)$
- $L_0$：最后一步的重构

---

## §5 处理 $L_{t-1}$（核心）

### 5.1 $q(x_{t-1} | x_t, x_0)$ 已知

由 derive_02：
$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(\tilde\mu_t(x_t, x_0), \tilde\beta_t I)$$

### 5.2 参数化 $p_\theta(x_{t-1} | x_t)$

DDPM 假设：
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 I)$$

其中：
- $\mu_\theta$：网络预测的均值
- $\sigma_t^2$：固定方差（$\beta_t$ 或 $\tilde\beta_t$，不学）

### 5.3 两个等方差高斯之间的 KL

**事实**：对 $p_1 = \mathcal{N}(\mu_1, \sigma^2 I)$ 和 $p_2 = \mathcal{N}(\mu_2, \sigma^2 I)$：
$$D_{\mathrm{KL}}(p_1 \| p_2) = \frac{\|\mu_1 - \mu_2\|^2}{2\sigma^2}$$

（推导：高斯 KL 通用公式在等方差下大幅简化）

但 DDPM 中实际上是 $q$ 用 $\tilde\beta_t$，$p_\theta$ 用 $\sigma_t^2$。**当 $\sigma_t^2 = \tilde\beta_t$** 时：

$$L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\tilde\beta_t}\|\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2\right] + \text{const}$$

---

## §6 进一步参数化：噪声预测

### 6.1 重参数化

由 derive_02 §3.3：
$$\tilde\mu_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \cdot \epsilon\right)$$

其中 $\epsilon$ 是用来从 $x_0$ 加噪到 $x_t$ 的噪声：$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$。

---

### 6.2 DDPM 选择匹配形式的参数化

DDPM 的灵感：**让 $\mu_\theta$ 模仿 $\tilde\mu_t$ 的形式**，只把 $\epsilon$ 换成网络预测 $\epsilon_\theta(x_t, t)$：

$$\mu_\theta(x_t, t) := \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \cdot \epsilon_\theta(x_t, t)\right)$$

这样：
$$\tilde\mu_t - \mu_\theta = \frac{1}{\sqrt{\alpha_t}} \cdot \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \cdot (\epsilon_\theta - \epsilon)$$

模长平方：
$$\|\tilde\mu_t - \mu_\theta\|^2 = \frac{1}{\alpha_t} \cdot \frac{\beta_t^2}{1-\bar\alpha_t} \cdot \|\epsilon - \epsilon_\theta\|^2$$

---

### 6.3 代入 $L_{t-1}$

$$L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\tilde\beta_t} \cdot \frac{1}{\alpha_t} \cdot \frac{\beta_t^2}{1-\bar\alpha_t} \cdot \|\epsilon - \epsilon_\theta\|^2\right]$$

整理常数（记为 $w_t$）：
$$L_{t-1} = w_t \cdot \mathbb{E}_{x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

其中 $w_t = \frac{\beta_t^2}{2 \tilde\beta_t \alpha_t (1-\bar\alpha_t)}$。

---

## §7 简化目标：去除时间权重

### 7.1 DDPM 的实验发现

Ho 2020 经验性发现：**去掉 $w_t$，对所有 $t$ 用等权重 MSE，效果反而更好**：

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t \sim U[1, T], x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

其中 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$。

---

### 7.2 为什么去掉权重更好？

仔细看 $w_t$：

$$w_t = \frac{\beta_t^2}{2 \tilde\beta_t \alpha_t (1-\bar\alpha_t)}$$

- **小 $t$**（$\beta_t$ 小，$1-\bar\alpha_t$ 小）：$w_t$ 大
- **大 $t$**（$\beta_t$ 大，$1-\bar\alpha_t$ 近 1）：$w_t$ 较小

带权重的目标过分关注小 $t$（容易的任务），忽略大 $t$（困难的任务）。

去掉权重相当于给所有时间步**等量训练资源**——大 $t$ 区域学得更好，最终生成质量提升。

---

### 7.3 最终公式（整个 DDPM 训练逻辑）

$$\boxed{\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t \sim U[1, T], \, x_0 \sim p_{\text{data}}, \, \epsilon \sim \mathcal{N}(0, I)}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon, t)\|^2\right]}$$

—— **简洁的 MSE 回归**。背后是 ELBO + 高斯 KL + 重参数化 + 适当的网络参数化的整套机器。

---

## §8 关于 $L_0$（最后一步）

DDPM 对 $p_\theta(x_0 | x_1)$ 有专门处理（连续像素值 → 离散像素值的 likelihood）。

Ho 2020 实践：当 $t=1$ 时，$L_0$ 仍可以归并入 simplified loss 形式（即 $t$ 从 1 开始也用 MSE）。最终对生成质量影响微小。

具体细节可参考 Ho 2020 论文 §3.3。

---

## §9 关于 $L_T$（终态）

$L_T = D_{\mathrm{KL}}(q(x_T | x_0) \| p(x_T)) = D_{\mathrm{KL}}(\mathcal{N}(\sqrt{\bar\alpha_T} x_0, (1-\bar\alpha_T) I) \| \mathcal{N}(0, I))$

由于 $\bar\alpha_T \approx 0$（schedule 设计如此），$\sqrt{\bar\alpha_T} x_0 \approx 0$、$1-\bar\alpha_T \approx 1$，所以 $q(x_T | x_0) \approx p(x_T)$，$L_T \approx 0$。

**不含 $\theta$**，训练时完全可以忽略。

---

## §10 Hybrid Loss（Improved DDPM）

Nichol & Dhariwal 2021 提出 hybrid loss：

$$\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{\text{simple}} + \lambda \cdot \mathcal{L}_{\text{vlb}}$$

其中 $\mathcal{L}_{\text{vlb}}$ 是上面推导的完整 ELBO（不去权重，带 $L_T, L_{t-1}, L_0$）。

**作用**：
- $\mathcal{L}_{\text{simple}}$ 提供主梯度信号（噪声预测）
- $\mathcal{L}_{\text{vlb}}$ 让方差网络（learned variance）也有梯度

$\lambda = 0.001$（小权重），避免 vlb 干扰 noise prediction。

---

## §11 验证：噪声预测 ≡ score 估计

由 derive_02 定理一，$q(x_t | x_0)$ 是高斯，所以：
$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{x_t - \sqrt{\bar\alpha_t} x_0}{1 - \bar\alpha_t} = -\frac{\sqrt{1-\bar\alpha_t} \epsilon}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

—— **预测 $\epsilon$ ≡ 预测 score**，差一个时间相关的系数。

这是 DDPM 与 score matching 在数学上严格等价的证明。

---

## §12 自查题

1. 完整复述 ELBO 的三部分分解
2. 解释为什么 $L_T$ 不影响训练
3. 推导 $\tilde\mu_t - \mu_\theta$ 的简化形式
4. 解释"去除权重"为什么效果更好

---

## §13 参考文献

- Ho et al., *DDPM*, NeurIPS 2020, §3
- Nichol & Dhariwal, *Improved DDPM*, ICML 2021
- Luo, *Understanding Diffusion Models: A Unified Perspective*, 2022（极佳的教学材料）
