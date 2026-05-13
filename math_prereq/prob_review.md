# 概率论速查 · 扩散模型必备

> 本材料**只覆盖扩散模型课程必需**的概率知识点，不是完整的概率论复习。
> 阅读节奏：每个小节配一个"扩散模型中如何用到"的提示，理解工具与应用的对应。
> 预计阅读时间：1.5–2 小时。

---

## §1 高斯分布与其线性性质

### 1.1 一维高斯

概率密度函数：

$$p(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

记号：$x \sim \mathcal{N}(\mu, \sigma^2)$。

**两个数字描述一切**：高斯分布完全由均值和方差决定。

---

### 1.2 高斯的线性变换（极其重要）

**事实 1**：若 $X \sim \mathcal{N}(\mu, \sigma^2)$，$Y = aX + b$，则
$$Y \sim \mathcal{N}(a\mu + b, a^2 \sigma^2)$$

**事实 2**：若 $X_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$、$X_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$、$X_1 \perp X_2$，则
$$X_1 + X_2 \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$$

**事实 3**：高斯的高斯还是高斯。即条件分布、边缘分布、联合分布都保持高斯性。

> **🔑 在扩散模型中**：DDPM 把图像加噪过程定义为一系列高斯转移。事实 1+2 让我们能把"$T$ 步加噪"合并成"一步从 $x_0$ 到 $x_T$ 的高斯采样"——这就是 $q(x_t | x_0)$ 闭合形式的核心来源。

---

### 1.3 重参数化（reparameterization trick）

**问题**：我们想从 $\mathcal{N}(\mu, \sigma^2)$ 中采样 $x$，但希望梯度能流回 $\mu$ 和 $\sigma$。

**直接采样不可导**：`torch.normal(mu, sigma)` 是一个不连续的随机操作，反向传播无法穿过它。

**解决**：把随机性"外包"给一个固定分布，参数仅做确定性映射：

$$x = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

现在 $\mu$ 和 $\sigma$ 与 $x$ 之间是**确定性**的链条，梯度可以正常流过。

**多维情况**：若想从 $\mathcal{N}(\mu, \Sigma)$ 采样，可以写成
$$x = \mu + L \epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad L L^\top = \Sigma$$

> **🔑 在扩散模型中**：DDPM 的训练 loss 中含有 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$ —— 这正是重参数化的应用。这个等式让我们可以"一步到位"采样到任意时间步 $t$。

---

### 1.4 多维高斯

对于 $x \in \mathbb{R}^d$，多维高斯：

$$p(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x-\mu)^\top \Sigma^{-1} (x-\mu)\right)$$

记号：$x \sim \mathcal{N}(\mu, \Sigma)$。

**特殊情况**：当 $\Sigma = \sigma^2 I$（各维独立同方差），密度退化为：

$$p(x) = \frac{1}{(2\pi\sigma^2)^{d/2}} \exp\left(-\frac{\|x-\mu\|^2}{2\sigma^2}\right)$$

> **🔑 在扩散模型中**：图像每个像素被独立加入相同方差的噪声，这就是 $\Sigma = \sigma^2 I$ 的情况。这让所有计算都退化为逐像素操作。

---

## §2 KL 散度

### 2.1 定义

对于两个概率分布 $p(x)$ 和 $q(x)$：

$$D_{\mathrm{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} \mathrm{d}x = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]$$

**直觉解读**：当我们用 $q$ 来近似真实分布 $p$ 时所付出的"额外信息代价"。

---

### 2.2 关键性质

**性质 1（非负性）**：$D_{\mathrm{KL}}(p \| q) \geq 0$，等号当且仅当 $p = q$ 几乎处处成立。

**性质 2（不对称）**：$D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)$。

不对称带来不同的"近似策略"：
- $D_{\mathrm{KL}}(p \| q)$（**forward KL**）：让 $q$ 覆盖 $p$ 的所有支撑（mode-covering）
- $D_{\mathrm{KL}}(q \| p)$（**reverse KL**）：让 $q$ 锁定 $p$ 的某个 mode（mode-seeking）

---

### 2.3 高斯之间的 KL 散度（必背公式）

两个一维高斯 $\mathcal{N}(\mu_1, \sigma_1^2)$ 和 $\mathcal{N}(\mu_2, \sigma_2^2)$：

$$D_{\mathrm{KL}}\bigl(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)\bigr) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

**特例**：当 $\sigma_1 = \sigma_2 = \sigma$，化简为
$$D_{\mathrm{KL}} = \frac{(\mu_1 - \mu_2)^2}{2\sigma^2}$$

—— 这是一个**带权重的 MSE**！这是 DDPM 训练目标可以简化为 MSE 的根源。

> **🔑 在扩散模型中**：DDPM 的 ELBO 可以写成多个高斯之间 KL 的求和。利用上述公式，每个 KL 项都化为关于均值的 MSE，最终简化成噪声预测的 MSE loss。

---

## §3 期望、条件期望与边缘化

### 3.1 全期望公式

$$\mathbb{E}_X[g(X)] = \mathbb{E}_Y\bigl[\mathbb{E}_X[g(X) | Y]\bigr]$$

直觉：把任何期望都可以"按某个变量分层"再组合。

> **🔑 在扩散模型中**：经常需要把 $\mathbb{E}_{x_t}$ 拆成 $\mathbb{E}_{x_0} \mathbb{E}_{x_t | x_0}$，因为 $q(x_t | x_0)$ 是闭合可采样的。

---

### 3.2 边缘化

$$p(x) = \int p(x, z) \mathrm{d}z = \int p(x | z) p(z) \mathrm{d}z$$

—— 把不关心的随机变量"积分掉"。

> **🔑 在扩散模型中**：模型最终关心的是 $p_\theta(x_0)$，但训练时通过引入辅助变量 $x_1, \dots, x_T$ 让问题变得可处理（边缘化掉这些变量后才得到 $x_0$ 的分布）。

---

### 3.3 蒙特卡洛估计

期望 $\mathbb{E}_{x \sim p}[f(x)]$ 通过样本估计：

$$\mathbb{E}_{x \sim p}[f(x)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i), \quad x_i \sim p$$

**误差**：标准差按 $1/\sqrt{N}$ 衰减，与维度无关（这是 MC 方法的优势）。

> **🔑 在扩散模型中**：训练时 loss 是对所有时间步 $t$ 和所有样本的期望，实际操作是**每个 batch 随机采样 $t$**——这就是单样本 MC 估计。

---

## §4 贝叶斯框架

### 4.1 贝叶斯公式

$$p(\theta | x) = \frac{p(x | \theta) \cdot p(\theta)}{p(x)}$$

- $p(\theta)$：先验
- $p(x | \theta)$：似然
- $p(\theta | x)$：后验
- $p(x) = \int p(x | \theta) p(\theta) \mathrm{d}\theta$：证据（evidence）

---

### 4.2 后验的高斯共轭

如果先验和似然都是高斯，后验也是高斯，且参数有闭合公式。这是扩散模型反向过程"为什么是高斯"的根源。

**重要事实**：DDPM 中的 $q(x_{t-1} | x_t, x_0)$（已知 $x_0$ 时的反向后验）是闭合高斯，而 $q(x_{t-1} | x_t)$（未知 $x_0$ 时）不是。所以训练时模型实际是在拟合前者。

---

## §5 变分推断与 ELBO

### 5.1 问题设置

我们想最大化数据的对数似然 $\log p_\theta(x)$。但当存在隐变量 $z$ 时，

$$\log p_\theta(x) = \log \int p_\theta(x, z) \mathrm{d}z$$

这个积分通常无法解析或高效近似。

---

### 5.2 ELBO 推导（关键技巧）

引入任意分布 $q(z | x)$，做如下变换：

$$
\begin{aligned}
\log p_\theta(x) &= \log \int p_\theta(x, z) \mathrm{d}z \\
&= \log \int q(z|x) \cdot \frac{p_\theta(x, z)}{q(z|x)} \mathrm{d}z \\
&= \log \mathbb{E}_{q}\left[\frac{p_\theta(x, z)}{q(z|x)}\right] \\
&\geq \mathbb{E}_{q}\left[\log \frac{p_\theta(x, z)}{q(z|x)}\right] \quad (\text{Jensen 不等式}) \\
&= \underbrace{\mathbb{E}_{q}[\log p_\theta(x, z)] - \mathbb{E}_{q}[\log q(z|x)]}_{\text{ELBO}}
\end{aligned}
$$

**ELBO 的等价形式**（更常见的"重构 + 正则"形式）：

$$\mathrm{ELBO}(x) = \mathbb{E}_{q(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q(z|x) \| p(z))$$

- 第一项：重构项（让 $q$ 编码下能重构出 $x$）
- 第二项：正则项（让后验 $q$ 接近先验 $p(z)$）

---

### 5.3 ELBO 与真实对数似然的差距

$$\log p_\theta(x) - \mathrm{ELBO}(x) = D_{\mathrm{KL}}(q(z|x) \| p_\theta(z|x)) \geq 0$$

— 差距正好是变分后验和真实后验的 KL 散度。

**含义**：
- 最大化 ELBO ≡ 同时做两件事：提升真实似然 + 让 $q$ 逼近真实后验
- 当 $q$ 完全等于真实后验时，ELBO 与 $\log p$ 完全相等

> **🔑 在扩散模型中**：DDPM 的训练目标就是一个 $T$ 步隐变量模型的 ELBO。把 $q(x_{1:T} | x_0)$ 设为固定的加噪过程，然后让 $p_\theta(x_{0:T})$ 学着反过来生成——这就把生成问题转化成了分步去噪问题。

---

## §6 Score Function（得分函数）

### 6.1 定义

$$s(x) = \nabla_x \log p(x)$$

**几何直觉**：score 是"指向高密度方向"的向量场。

如果你站在 $x$ 处想往密度更高的地方走，沿 score 方向走一小步即可。

---

### 6.2 高斯的 score

对 $p(x) = \mathcal{N}(\mu, \sigma^2 I)$：

$$\nabla_x \log p(x) = -\frac{x - \mu}{\sigma^2}$$

—— 指向均值方向，距离均值越远 score 越大。

---

### 6.3 Score 与 noise prediction 的等价性

DDPM 中，$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$。

可以证明：
$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

—— **预测 score 和预测 noise 等价**，差一个时间相关的系数。

> **🔑 在扩散模型中**：这是统一 DDPM（noise prediction）和 score-based 模型（score prediction）的关键事实。两条研究线最终在 Score SDE（Song 2021）中合流。

---

## §7 一些特殊技巧

### 7.1 Log-derivative trick

$$\nabla_\theta \log p_\theta(x) = \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)}$$

—— 对数 + 分数 ≡ 梯度。这在 score matching、policy gradient 中反复出现。

---

### 7.2 Stein's identity（看不懂可跳过）

对任何足够好的函数 $\phi$：
$$\mathbb{E}_{x \sim p}[\nabla_x \log p(x) \cdot \phi(x) + \nabla_x \phi(x)] = 0$$

这是 score matching 的核心理论基础（让 score 估计避开了求归一化常数 $Z$ 的问题）。

---

## §8 自查清单

读完本材料后，你应该能不查公式回答下列问题：

- [ ] 写出一维和多维高斯的密度函数
- [ ] 写出高斯线性变换后的均值方差
- [ ] 写出重参数化采样表达式
- [ ] 写出 KL 散度的定义
- [ ] 写出两个等方差高斯的 KL 化简结果
- [ ] 一句话说出 ELBO 是什么
- [ ] 写出 ELBO 的两种等价形式
- [ ] 写出 score function 的定义
- [ ] 解释为什么 score 和 noise 在 DDPM 中等价

如果某些项答不出，回到对应小节再读一遍。

---

## §9 推荐进一步阅读

| 资源 | 适合人群 | 重点 |
|------|----------|------|
| Bishop《PRML》Chapter 2 | 深入推导 | 高斯、共轭分布的完整数学 |
| Murphy《Probabilistic ML》Chapter 1–4 | 现代视角 | 与机器学习紧密结合 |
| Lilian Weng 博客 "From Autoencoder to Beta-VAE" | 直觉理解 | ELBO 的另一种讲法 |
| 3Blue1Brown 贝叶斯视频 | 几何直觉 | 培养直觉用 |
