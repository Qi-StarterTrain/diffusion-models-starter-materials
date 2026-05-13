# 数学前置自测 · 参考答案

> 自测完成后再来对照本答案。每题附简要解释，错题应回到相应章节复习。

---

## 第一部分：概率与统计

### 题 1 答案

$Y \sim \mathcal{N}(a\mu + b, a^2 \sigma^2)$

**关键点**：高斯分布的线性变换仍是高斯，均值线性变化、方差按平方放缩。这是 DDPM 中所有"加噪推导"的基础工具。

→ 不会请复习：`prob_review.md` §1.2 高斯分布性质

---

### 题 2 答案

$x = \mu + \sigma \cdot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, 1)$。

**关键点**：这就是 **reparameterization trick**。
- 直接采样 $x \sim \mathcal{N}(\mu, \sigma^2)$ 不可导（采样是不连续操作）
- 把随机性"外包"给 $\epsilon$，则 $\mu, \sigma$ 与 $x$ 之间是确定性映射，可反向传播

VAE、扩散模型的训练全部依赖这个技巧。

→ 不会请复习：`prob_review.md` §1.3 重参数化

---

### 题 3 答案

$X_1 + X_2 \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$

**关键点**：独立高斯之和仍是高斯，均值/方差均相加。
DDPM 中"两次加噪可合并为一次"的推导基于此。

---

### 题 4 答案

$$D_{\mathrm{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} \mathrm{d}x$$

性质：
1. **非负性**：$D_{\mathrm{KL}}(p \| q) \geq 0$，等号当且仅当 $p = q$ 几乎处处成立
2. **不对称**：$D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)$（一般情况下）

**关键点**：KL 是 ELBO 推导的核心工具，理解它的非对称性能帮助理解为什么 forward KL 和 reverse KL 在 VAE/扩散中的作用不同。

→ 不会请复习：`prob_review.md` §2 KL 散度

---

### 题 5 答案

$$p(\theta | x) = \frac{p(x | \theta) \cdot p(\theta)}{p(x)}$$

- $p(\theta)$：**先验**，参数本身的分布
- $p(x | \theta)$：**似然**，给定参数下数据的概率
- $p(\theta | x)$：**后验**，看到数据后对参数的更新

**关键点**：扩散模型的 reverse process $p_\theta(x_{t-1} | x_t)$ 本质上是在估计后验 $q(x_{t-1} | x_t, x_0)$ 的近似。

---

## 第二部分：微积分与线性代数

### 题 6 答案

$$\frac{\partial L}{\partial x} = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$$

**关键点**：链式法则是反向传播的本质。所有深度学习框架都是基于这个简单公式构建的自动微分系统。

---

### 题 7 答案

$\nabla f = (2x + 3y, 3x + 2y)$

在 $(1, 2)$ 处：$\nabla f(1, 2) = (2 + 6, 3 + 4) = (8, 7)$

---

### 题 8 答案

**是**。

推导：设 $f = x^\top A x = \sum_i \sum_j A_{ij} x_i x_j$

$$\frac{\partial f}{\partial x_k} = \sum_j A_{kj} x_j + \sum_i A_{ik} x_i = (Ax)_k + (A^\top x)_k$$

故 $\frac{\partial f}{\partial x} = (A + A^\top) x$。

**特例**：当 $A$ 对称时，$\frac{\partial f}{\partial x} = 2Ax$（这是高斯密度推导中的常见情况）。

---

### 题 9 答案

$\int_{-\infty}^{+\infty} \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \mathrm{d}x = 1$

**关键点**：这是高斯分布的归一化常数。如果你"心里没有这个 1"，做扩散模型推导会非常痛苦。

---

## 第三部分：深度学习与 PyTorch

### 题 10 答案

**B 和 D**

- A 错误：反向传播是按**逆序**计算
- B 正确：链式法则的本质就是从后往前
- C 错误：默认对所有 `requires_grad=True` 的张量都计算梯度
- D 正确：这就是反向传播显存开销的来源（也是 gradient checkpointing 要解决的问题）

---

### 题 11 答案

```
y.shape = torch.Size([4, 3])
y.is_contiguous() = False
```

**关键点**：`permute` 只改变 stride，不改变底层内存布局。如果之后要做 `view`，需要先 `.contiguous()`。这是 PyTorch 中最常踩的坑之一。

---

### 题 12 答案

**MSE Loss（均方误差）**

$$L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

**关键点**：DDPM 的简化训练目标本质上就是噪声的 MSE。理解这一点后会发现 DDPM 的训练 loss 极其简单。

---

### 题 13 答案

1. **一阶动量（momentum）**：使用梯度的指数移动平均，$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$，作用是平滑梯度方向、加速收敛
2. **二阶动量（adaptive learning rate）**：使用梯度平方的指数移动平均，$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$，作用是为每个参数自适应学习率（梯度大的参数学习率自动减小）

最终更新：$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

---

### 题 14 答案

**参考答案**：U-Net 是一种编码器-解码器结构的卷积网络，包含下采样路径（编码器，逐步降低分辨率提取语义信息）和上采样路径（解码器，逐步恢复分辨率），并在对应分辨率层之间通过 **skip connection** 直接连接，使得低层细节信息可以直接传递到解码器。

**关键点**：
- skip connection 在扩散 U-Net 中至关重要（去噪需要细节信息）
- 时间步 embedding 通过广播加法注入到每个 ResBlock
- attention 通常只加在低分辨率层（计算量考虑）

---

## 第四部分：综合应用

### 题 15 答案

**(a)** 由于 $\epsilon \sim \mathcal{N}(0, I)$，$x_0$ 是确定的：
- 均值：$E[x_1] = \sqrt{1-\beta} \cdot x_0$
- 方差：$\mathrm{Var}[x_1] = \beta \cdot I$

**(b)** $x_1 | x_0 \sim \mathcal{N}(\sqrt{1-\beta} \cdot x_0, \beta \cdot I)$

**(c)** $\mathcal{L} = \mathbb{E}_{x_0, \epsilon} \left[ \| f_\theta(x_1) - x_0 \|^2 \right]$

或等价地，预测噪声形式：$\mathcal{L} = \mathbb{E}_{x_0, \epsilon} \left[ \| g_\theta(x_1) - \epsilon \|^2 \right]$

**(d) 关键洞察**：
- 加噪过程是一个固定的、可解析计算的高斯转移；
- 去噪过程通过神经网络学习；
- 如果把这个 1 步过程扩展到 T 步，使加噪逐渐进行直到完全噪声，再让网络学习反向去噪，就得到了 DDPM 的形式。

**这其实就是 DDPM 的雏形！** 课程第 2 周会把这个过程严格化、推广到 T 步、并推导出可训练的 loss 形式。

---

## 综合评分

| 部分 | 满分 | 自评得分 | 备注 |
|------|------|----------|------|
| 概率与统计 | 5 | | 任何一题错都建议复习对应小节 |
| 微积分与线性代数 | 4 | | 所有题都应该是闭眼能做 |
| 深度学习与 PyTorch | 5 | | PyTorch 不熟练会严重拖慢实验 |
| 综合应用 | 1 | | 这题反映你是否能把基础知识"组装"起来 |
| **总分** | **15** | | |

---

## 最重要的几个补课资源

如果你某些题答错了，按优先级推荐以下补充材料：

**概率方向**
- Bishop《Pattern Recognition and Machine Learning》第 1–2 章
- 3Blue1Brown：贝叶斯定理直觉视频

**微积分 / 线代方向**
- 3Blue1Brown：线性代数本质系列
- MIT 18.06 Strang 线性代数

**PyTorch 方向**
- 官方教程 "60 minute blitz"
- PyTorch examples 仓库中的 `mnist` 和 `dcgan`

**深度学习基础**
- d2l.ai《动手学深度学习》前 5 章
