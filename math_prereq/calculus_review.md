# 微积分与线代速查 · 扩散模型必备

> 只覆盖扩散模型课程必需的微积分与线性代数知识。
> 预计阅读时间：1–1.5 小时。

---

## §1 链式法则（反向传播的灵魂）

### 1.1 单变量链式法则

设 $L = f(g(h(x)))$，$x \in \mathbb{R}$，则：

$$\frac{\partial L}{\partial x} = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$$

—— 沿计算路径依次求导，再连乘。

---

### 1.2 多变量链式法则

设 $L = f(y_1, y_2, \dots, y_n)$，每个 $y_i = g_i(x)$，则：

$$\frac{\partial L}{\partial x} = \sum_{i=1}^n \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial x}$$

—— 多条路径上的梯度**求和**。

> **🔑 在扩散模型中**：U-Net 中 skip connection 让一个张量同时进入下采样和上采样路径，反向传播时两条路径的梯度相加——这正是上述公式的应用。

---

### 1.3 矩阵情况

设 $y = f(x)$，$x \in \mathbb{R}^n$，$y \in \mathbb{R}^m$。

雅可比矩阵 $J \in \mathbb{R}^{m \times n}$：$J_{ij} = \frac{\partial y_i}{\partial x_j}$。

链式法则向量形式：
$$\frac{\partial L}{\partial x} = J^\top \cdot \frac{\partial L}{\partial y}$$

—— PyTorch 反向传播的本质就是反复做这个 vector-Jacobian 乘积。

---

## §2 梯度

### 2.1 定义

对 $f : \mathbb{R}^n \to \mathbb{R}$：

$$\nabla f(x) = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)$$

**几何意义**：$\nabla f(x)$ 指向 $f$ 在 $x$ 处增长最快的方向；模长是该方向的变化率。

---

### 2.2 梯度下降

最小化 $f$ 的迭代：
$$x_{t+1} = x_t - \eta \nabla f(x_t)$$

其中 $\eta$ 是学习率。

**几何理解**：每一步沿当前梯度的反方向走一小步——朝下坡走。

---

### 2.3 常用梯度公式（**必背**）

| 表达式 | 对 $x$ 的梯度 |
|--------|-------------|
| $a^\top x$ | $a$ |
| $x^\top A x$（$A$ 一般） | $(A + A^\top) x$ |
| $x^\top A x$（$A$ 对称） | $2Ax$ |
| $\| x - a \|^2$ | $2(x - a)$ |
| $\log \det(X)$（$X$ 对称正定） | $X^{-1}$ |
| $\text{tr}(AX)$（对 $X$） | $A^\top$ |

---

## §3 积分（必要技巧）

### 3.1 高斯积分（**必记**）

$$\int_{-\infty}^{+\infty} e^{-x^2/2} \mathrm{d}x = \sqrt{2\pi}$$

更一般地：
$$\int_{-\infty}^{+\infty} \frac{1}{\sqrt{2\pi}\sigma} e^{-(x-\mu)^2/(2\sigma^2)} \mathrm{d}x = 1$$

—— 高斯密度的归一化条件。

---

### 3.2 换元积分

$$\int f(g(x)) g'(x) \mathrm{d}x = \int f(u) \mathrm{d}u, \quad u = g(x)$$

**雅可比版（多维）**：若 $u = T(x)$ 是可逆变换，$x \in \mathbb{R}^n$：

$$\int f(u) \mathrm{d}u = \int f(T(x)) \cdot |\det J_T(x)| \mathrm{d}x$$

> **🔑 在 normalizing flow 和 diffusion 的 ODE 视角中**：变换前后的密度通过雅可比行列式联系起来——这是连续时间生成模型的核心数学。

---

## §4 SDE / ODE 简介（Score SDE 课程必备）

### 4.1 ODE（常微分方程）

形式：$\frac{\mathrm{d} x}{\mathrm{d} t} = f(x, t)$

**理解**：给定一个向量场 $f(x, t)$，从某个初始点 $x_0$ 出发，沿向量场走，得到一条轨迹 $x(t)$。

**数值求解**：欧拉法 $x_{t+\Delta t} = x_t + f(x_t, t) \cdot \Delta t$。

> **🔑 在扩散模型中**：DDIM 和 probability flow ODE 都把扩散过程视为一个 ODE，sampler 就是数值求解器。DPM-Solver 等现代 sampler 本质是高阶 ODE 求解器。

---

### 4.2 SDE（随机微分方程）

形式：$\mathrm{d} x = f(x, t) \mathrm{d} t + g(t) \mathrm{d} W$

- $f(x, t)$：drift（确定性方向）
- $g(t)$：diffusion coefficient（噪声强度）
- $\mathrm{d} W$：Wiener 过程的"无穷小增量"，可粗略理解为 $\mathcal{N}(0, \mathrm{d} t)$

**数值近似**：$x_{t+\Delta t} = x_t + f(x_t, t) \Delta t + g(t) \sqrt{\Delta t} \cdot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, 1)$。

> **🔑 在扩散模型中**：DDPM 的离散加噪过程在连续极限下就是一个 SDE。Score SDE（Song 2021）建立了 DDPM ↔ SDE 的对应。

---

### 4.3 概率流 ODE

每个 SDE 都有一个对应的"概率流 ODE"，它在每个时刻 $t$ 的边缘分布与原 SDE 完全相同，但**轨迹是确定性的**。

形式：
$$\frac{\mathrm{d} x}{\mathrm{d} t} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$$

**意义**：去随机性的扩散等价物。这让我们可以用 ODE 高阶求解器进行确定性、可逆的采样。

---

## §5 线性代数速查

### 5.1 矩阵乘法的形状

设 $A \in \mathbb{R}^{m \times n}$，$B \in \mathbb{R}^{n \times p}$，则 $AB \in \mathbb{R}^{m \times p}$。

—— **内层维度必须匹配**。

---

### 5.2 转置、逆、行列式

- $(AB)^\top = B^\top A^\top$
- $(AB)^{-1} = B^{-1} A^{-1}$（前提：都可逆）
- $\det(AB) = \det(A) \det(B)$

---

### 5.3 特征值与特征向量

$Av = \lambda v$（$v \neq 0$）

- $\lambda$：特征值
- $v$：特征向量

对**对称矩阵**，特征向量正交，特征值实数；可正交对角化为 $A = Q \Lambda Q^\top$。

---

### 5.4 协方差矩阵

设 $X = (X_1, \dots, X_n)^\top$ 是随机向量，协方差：

$$\Sigma_{ij} = \mathrm{Cov}(X_i, X_j) = \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)]$$

**性质**：
- 对称：$\Sigma = \Sigma^\top$
- 半正定：$v^\top \Sigma v \geq 0$ 对任意 $v$ 成立
- 对角元是各分量的方差

---

### 5.5 矩阵求导（仅列必备）

设 $f : \mathbb{R}^{m \times n} \to \mathbb{R}$，则 $\frac{\partial f}{\partial X} \in \mathbb{R}^{m \times n}$。

| 函数 | 对 $X$ 的梯度 |
|------|-------------|
| $\text{tr}(AX)$ | $A^\top$ |
| $\text{tr}(X^\top A X)$ | $(A + A^\top) X$ |
| $\log \det(X)$（正定） | $X^{-1}$ |

---

## §6 数值稳定性（深度学习常踩坑点）

### 6.1 Log-Sum-Exp trick

直接计算 $\log \sum_i e^{x_i}$ 在 $x_i$ 很大时会上溢，很小时会损失精度。

正确做法：
$$\log \sum_i e^{x_i} = m + \log \sum_i e^{x_i - m}, \quad m = \max_i x_i$$

> **🔑 在扩散模型中**：variational lower bound 计算、softmax loss 都隐含使用了这个技巧。

---

### 6.2 数值化的 reparameterization

$$\log \sigma \to \text{predict in log space, then exp}$$

让模型预测 $\log \sigma$ 而非 $\sigma$，避免 $\sigma \leq 0$ 的非法值。

> **🔑 在扩散模型中**：Improved DDPM 的 learned variance 就是用 $v$ 在 $[\beta_t, \tilde\beta_t]$ 之间插值，本质上是在做对数空间的预测。

---

### 6.3 fp16 的陷阱

fp16 表示范围约为 $[6 \times 10^{-5}, 6.5 \times 10^4]$。

- **下溢**：很小的值会被截断为 0（梯度消失）
- **上溢**：很大的值变成 inf（gradient explosion）
- **解决**：使用 `torch.cuda.amp` 的 GradScaler，或对关键层（GroupNorm、Softmax）保持 fp32

> **🔑 在扩散模型中**：U-Net 中的 GroupNorm 在 fp16 下容易出 NaN，是常见踩坑点。

---

## §7 自查清单

- [ ] 能流利写出多变量链式法则
- [ ] 知道梯度的方向意义
- [ ] 记得高斯积分的归一化结果
- [ ] 会做 $x^\top A x$ 求导
- [ ] 能解释 ODE 的几何意义
- [ ] 知道 fp16 训练时哪些层要保持 fp32
- [ ] 能写出 log-sum-exp 数值稳定形式

---

## §8 推荐资源

| 资源 | 重点 |
|------|------|
| 3Blue1Brown 微积分本质系列 | 几何直觉 |
| 3Blue1Brown 线性代数本质系列 | 几何直觉 |
| Goodfellow《Deep Learning》第 2 章 | 深度学习视角的线代/微积分 |
| The Matrix Cookbook（PDF） | 矩阵求导查表手册 |
