# Lecture 02｜数学前置：VAE 与 Score Matching

> **本讲目标**
> - 通过 VAE 完整推导建立"用 ELBO 训练隐变量模型"的范式（DDPM 是其延伸）
> - 理解 score function 的几何意义与 score matching 的训练目标
> - 看到这两条线（ELBO + score）如何最终在 Diffusion 中合流
> - 配齐后续课程必需的所有数学工具

---

## §1 VAE 完整推导（DDPM 的"前传"）

### 1.1 为什么从 VAE 讲起？

VAE 与 DDPM 在数学结构上**几乎一致**：
- 都是隐变量模型（latent variable model）
- 都最大化 ELBO
- 都用重参数化做训练

理解 VAE 后，DDPM 就是"把一个 latent $z$ 替换成 $T$ 个 $z_1, \dots, z_T$"。

---

### 1.2 隐变量模型设置

引入隐变量 $z$，定义联合分布：
$$p_\theta(x, z) = p_\theta(x | z) \cdot p(z)$$

其中：
- $p(z) = \mathcal{N}(0, I)$：先验，固定
- $p_\theta(x | z)$：解码器（decoder），由神经网络参数化

我们关心的边际似然：
$$p_\theta(x) = \int p_\theta(x | z) p(z) \mathrm{d}z$$

**问题**：这个积分一般无法解析、也无法高效采样估计（高维 $z$ 上 MC 估计效率极低）。

---

### 1.3 引入变分后验

引入近似后验 $q_\phi(z | x)$（编码器，encoder），用神经网络参数化为高斯：

$$q_\phi(z | x) = \mathcal{N}\bigl(z; \mu_\phi(x), \sigma_\phi^2(x) \cdot I\bigr)$$

**关键观察**：用 $q_\phi(z | x)$ 我们可以从 $z$ 的"有意义区域"采样，而不是盲目从 $p(z)$ 采样。

---

### 1.4 ELBO 推导（必须会手推）

我们的目标是最大化 $\log p_\theta(x)$。引入 $q_\phi(z|x)$：

$$
\begin{aligned}
\log p_\theta(x) &= \log \int p_\theta(x, z) \mathrm{d}z \\
&= \log \int q_\phi(z|x) \cdot \frac{p_\theta(x, z)}{q_\phi(z|x)} \mathrm{d}z \\
&= \log \mathbb{E}_{q_\phi(z|x)}\left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right] \\
&\geq \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right] \quad (\text{Jensen 不等式}) \\
&= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x | z)] - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))
\end{aligned}
$$

最终形式：

$$
\boxed{\log p_\theta(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x | z)]}_{\text{重构项}} - \underbrace{D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))}_{\text{正则项}}}
$$

**两项的物理含义**：
- **重构项**：$x$ 经编码—解码后能否重构得好？（让 $q_\phi$ 编码出的 $z$ 能被解码器还原 $x$）
- **正则项**：编码器输出的后验是否接近先验？（避免过拟合，保证 latent space 整齐）

---

### 1.5 ELBO 与真实对数似然的差距

可以严格证明：
$$\log p_\theta(x) - \mathrm{ELBO}(x) = D_{\mathrm{KL}}(q_\phi(z|x) \| p_\theta(z|x)) \geq 0$$

**意义**：
- 差距 = 变分后验和真实后验的 KL
- 当 $q_\phi$ 完全等于真实后验时，ELBO = 真实 likelihood
- 最大化 ELBO ≡ 同时**最大化对数似然 + 让 $q_\phi$ 逼近真实后验**

---

### 1.6 重参数化让训练可微

ELBO 中含 $\mathbb{E}_{q_\phi(z|x)}[\cdot]$，要计算梯度需对 $\phi$ 求导。但 $z \sim q_\phi$ 是采样操作，不可导。

**重参数化技巧**：
$$z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

把随机性外包给 $\epsilon$，则 $z$ 与 $\phi$ 之间是确定性映射，梯度可以正常流动。

---

### 1.7 实践中的 VAE 训练目标

设解码器为高斯：$p_\theta(x | z) = \mathcal{N}(\mu_\theta(z), I)$（固定单位方差）。

则重构项化简为：
$$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] = -\frac{1}{2} \mathbb{E}_{q_\phi}[\| x - \mu_\theta(z) \|^2] + \text{const}$$

—— **MSE 重构损失**！

正则项（两个高斯之间的 KL）有闭合公式：
$$D_{\mathrm{KL}}(\mathcal{N}(\mu, \sigma^2 I) \| \mathcal{N}(0, I)) = \frac{1}{2} \sum_i (\mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1)$$

最终训练目标：
$$\mathcal{L}_{\mathrm{VAE}} = \frac{1}{2} \| x - \mu_\theta(z) \|^2 + \frac{1}{2} \sum_i (\mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1)$$

—— 一个 MSE + 一个解析正则项，**普通监督学习的训练方式**就能训。

---

### 1.8 VAE 给 DDPM 的启示

| VAE 中 | DDPM 中 |
|--------|---------|
| 单一隐变量 $z$ | $T$ 个隐变量 $x_1, \dots, x_T$ |
| 编码器 $q_\phi(z\|x)$ | **固定**的加噪过程 $q(x_{1:T}\|x_0)$ |
| 解码器 $p_\theta(x\|z)$ | 学习的去噪过程 $p_\theta(x_{0:T})$ |
| 一步采样 | $T$ 步采样 |
| ELBO 训练 | ELBO 训练（最终化简为 MSE） |

**关键洞察**：DDPM 把 VAE 的"一个隐变量"扩展为"链式多个隐变量"，编码器固定为加噪，解码器学习反向去噪。

这就是为什么 DDPM 是隐变量生成模型大家庭的成员。

---

## §2 Score Function 与 Score Matching

### 2.1 Score function 的定义

对密度 $p(x)$，**得分函数**（score function）定义为：
$$s(x) = \nabla_x \log p(x)$$

注意：是对 $x$ 求导，不是对参数求导。

---

### 2.2 几何直觉

$\nabla_x \log p(x)$ 指向**密度更高的方向**，模长是 log-density 的局部变化率。

**类比**：把 $-\log p(x)$ 视为势能，则 $-s(x) = -\nabla_x \log p(x)$ 是力的方向；沿 $s(x)$ 方向走 = 沿"密度梯度"上坡。

---

### 2.3 Score 与采样：Langevin 动力学

如果你**有了真实分布的 score**，能否生成样本？

**可以**。Langevin 动力学迭代：
$$x_{t+1} = x_t + \frac{\eta}{2} s(x_t) + \sqrt{\eta} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

- $\eta$ 足够小、迭代足够多，$x_t$ 收敛到 $p$ 的样本
- 第一项：朝高密度方向走（drift）
- 第二项：随机扰动（保证遍历整个分布）

> **🔑 重要**：这意味着**只要会估计 score，就会做生成**。这是 score-based 模型的根本动机。

---

### 2.4 Score Matching：怎么估计未知分布的 score？

我们没有真实分布的 score，但有数据样本 $x_1, \dots, x_N \sim p_{\text{data}}$。

**目标**：训练一个网络 $s_\theta(x)$ 使其逼近真实的 $\nabla_x \log p_{\text{data}}(x)$。

**朴素想法**：最小化
$$\mathbb{E}_{x \sim p_{\text{data}}} \| s_\theta(x) - \nabla_x \log p_{\text{data}}(x) \|^2$$

—— 但我们**不知道**右边的真实 score！

---

### 2.5 Hyvärinen Score Matching（2005）

**重要结果**：可以证明上述目标等价于
$$\mathbb{E}_{x \sim p_{\text{data}}} \left[ \text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2} \| s_\theta(x) \|^2 \right]$$

这个目标**不需要真实 score**，只需要数据样本！

但实际中，$\text{tr}(\nabla_x s_\theta(x))$（雅可比矩阵的迹）在高维下计算极慢。

---

### 2.6 Denoising Score Matching（DSM）

**Vincent (2011)** 给出了一个高效近似：

不直接估计 $p_{\text{data}}$ 的 score，而是估计**加噪后**分布 $p_\sigma(\tilde x)$ 的 score。

设加噪 $\tilde x = x + \sigma \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$，则：

$$\nabla_{\tilde x} \log p_\sigma(\tilde x | x) = -\frac{\tilde x - x}{\sigma^2}$$

—— 已知 $x$ 时，加噪后样本的 score 有解析解！

DSM 训练目标：
$$\mathcal{L}_{\mathrm{DSM}} = \mathbb{E}_{x, \tilde x} \left[ \left\| s_\theta(\tilde x) - \nabla_{\tilde x} \log p_\sigma(\tilde x | x) \right\|^2 \right] = \mathbb{E}_{x, \epsilon} \left[ \left\| s_\theta(\tilde x) + \frac{\epsilon}{\sigma} \right\|^2 \right]$$

—— **本质是预测加进去的噪声**（差一个系数）。

> **🔑 这里就是与 DDPM 的连接点**：DDPM 训练网络预测噪声 $\epsilon$，等价于预测 score，差一个 $\sigma$ 系数。

---

### 2.7 多尺度 score matching：NCSN（Song & Ermon 2019）

单一噪声尺度 $\sigma$ 不够：
- $\sigma$ 太小：score 在低密度区域不准（数据流形外信息少）
- $\sigma$ 太大：丢失数据细节

**解决**：用一系列噪声尺度 $\sigma_1 > \sigma_2 > \cdots > \sigma_L$，训练**条件 score 网络** $s_\theta(x, \sigma)$。

采样时：从大噪声开始 Langevin 动力学，逐步减小 $\sigma$ 至 0。

> **🔑 这就是 DDPM 的"另一种血脉"**：从大噪声逐步采样到干净数据，本质上和 DDPM 的反向过程一致。Song 2021 的 Score SDE 论文最终把这两条线统一了。

---

## §3 两条线如何在 Diffusion 中合流

下表展示 VAE 路线（ELBO）和 Score 路线（DSM）如何分别得到等价的训练目标：

| 路线 | 起点 | 训练目标推导 | 最终形式 |
|------|------|------------|----------|
| **VAE → DDPM** | 最大化 $\log p_\theta(x_0)$ 的 ELBO | KL 化简、参数化技巧 | $\| \epsilon - \epsilon_\theta(x_t, t) \|^2$ |
| **Score Matching → NCSN** | 估计加噪分布 score | DSM 推导 | $\| \epsilon - \epsilon_\theta(x_t, t) \|^2$ |

—— **完全相同的训练目标，不同的推导路径**！

这是 Diffusion 模型理论之美的核心：从两个不同视角出发，**都得出"训练网络预测噪声"这个简单结论**。

下一讲（L03）我们将走完第一条路线（VAE → DDPM 的完整推导）。

---

## §4 概念与术语速查

| 术语 | 含义 | 课程后续用到 |
|------|------|-------------|
| ELBO | Evidence Lower Bound，对数似然下界 | DDPM 的 loss 推导 |
| 重构项 | ELBO 中的似然部分 | DDPM 的 $L_0$ 项 |
| 正则项 | ELBO 中的 KL 部分 | DDPM 的 $L_T$ 项 |
| 重参数化 | 把随机性"外包"使梯度可流 | DDPM 的 $x_t$ 闭合采样 |
| Score function | $\nabla_x \log p(x)$ | Score SDE，sampler 设计 |
| Score matching | 不知 ground truth score 时的估计方法 | NCSN，DSM |
| Langevin 动力学 | 用 score 采样的 SDE | Predictor-Corrector sampler |
| DSM | Denoising Score Matching | DDPM 的另一种推导 |

---

## §5 本讲核心要点回顾

1. **VAE 的 ELBO 推导是 DDPM 的数学母版**——重构项 + KL 正则 + 重参数化
2. **Score function 是另一条理解生成模型的路径**——只要能估计 score，就能用 Langevin 采样
3. **DSM 让我们能在加噪样本上训练 score 网络**，本质是预测噪声
4. **VAE 路线和 Score 路线最终给出相同的 DDPM 训练目标**
5. 下一讲我们走完 VAE → DDPM 的完整推导

---

## §6 课后任务

### 必做

1. **手推 ELBO**（提交手写或 LaTeX 推导）：
   - 从 $\log p_\theta(x) = \log \int p_\theta(x, z) \mathrm{d}z$ 出发
   - 引入 $q_\phi(z|x)$，使用 Jensen 不等式
   - 化简到"重构项 - KL 正则项"形式
   - 标注每一步使用的数学工具

2. **手推差距**：证明 $\log p_\theta(x) - \mathrm{ELBO}(x) = D_{\mathrm{KL}}(q_\phi(z|x) \| p_\theta(z|x))$。

3. **代码任务**：实现一个最小 VAE 在 MNIST 上训练
   - encoder/decoder 各 3 层 MLP
   - latent dim = 16
   - 训练 20 epoch，可视化生成样本与 latent space
   - 提交 `mnist_vae.ipynb` 与 reading note

4. **Reading note**：阅读 Vincent 2011《A Connection between Score Matching and Denoising Autoencoders》的核心定理，理解 DSM 的推导。

### 选做

5. 在 2D 玩具数据集（混合高斯）上实现 score matching 与 Langevin 采样，可视化 score field。可参考：[Yang Song 的 score-matching tutorial](https://yang-song.net/blog/2021/score/)

---

## §7 推荐进一步阅读

| 资源 | 重点 |
|------|------|
| Kingma & Welling 2013《Auto-Encoding Variational Bayes》 | VAE 原始论文 |
| Lilian Weng *From AE to Beta-VAE* | VAE 的另一种讲法 |
| Yang Song 博客 *Generative Modeling by Estimating Gradients of the Data Distribution* | Score-based 模型权威综述 |
| Vincent 2011 DSM 论文 | DSM 严格推导 |
| 3Blue1Brown KL 散度视频 | 几何直觉补充 |

---

## §8 常见问题

**Q: VAE 的 reconstruction loss 为什么常用 BCE 而不是 MSE？**

A: BCE 对应解码器为 Bernoulli（像素 ∈ {0,1}），MSE 对应高斯（像素 ∈ ℝ）。MNIST 等二值图像用 BCE 更自然，但严格来说应当根据数据分布选择。DDPM 中所有像素是连续值，所以全程用 MSE。

**Q: 重参数化为什么对**离散**变量不行？**

A: 重参数化要求"参数与样本的映射可微"，离散变量没有连续参数化，需要 Gumbel-Softmax 等近似。Diffusion 的所有变量都是连续的，这就是它特别适用于图像/视频/动作建模的原因之一。

**Q: Score matching 听起来比 ELBO 简单，为什么早期 NCSN 没击败 GAN？**

A: 单尺度 score matching 在数据流形外（低密度区域）估计极差，导致 Langevin 采样难以从噪声出发"找到"数据。NCSN 引入多尺度后才解决，但工程细节复杂。DDPM 用 ELBO + 噪声预测的形式让训练目标极其简洁，是工程胜利。

**Q: 这一讲的内容会被考试考到吗？**

A: 是的。ELBO 推导是**理论小测 1** 的重点。手推不熟练会严重影响你理解 DDPM 的训练目标。
