# Derivation 07: Flow Matching 与 Conditional FM 等价性

> 对应 L12 内容。本手稿严格证明：
> - Flow Matching loss 不可直接计算（要求未知的边缘 $u_t$）
> - Conditional Flow Matching loss **梯度等价**于 FM loss
> - 因此 CFM 可以训练（条件 $u_t(x|x_1)$ 解析可得）

---

## §1 设置与记号

- $t \in [0, 1]$：时间
- $p_0 = \mathcal{N}(0, I)$：先验
- $p_1 = p_{\text{data}}$：目标
- $\{p_t\}_{t \in [0,1]}$：连接 $p_0$ 与 $p_1$ 的**概率路径**
- $v_\theta(x, t)$：神经网络预测的向量场

我们希望训练 $v_\theta$ 使得 ODE $dx/dt = v_\theta(x, t)$ 沿路径 $p_t$ 传输。

---

## §2 Continuity Equation 与生成向量场

### 2.1 定义

向量场 $u_t(x)$ 称为**生成 $p_t$**，如果它满足 continuity equation：
$$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t u_t) = 0$$

直觉：$u_t$ 描述每个点的"流速"，使得密度按 $p_t$ 演化。

---

### 2.2 唯一性？

**不**。给定 $p_t$，可能有多个 $u_t$ 满足 continuity equation（差一个"divergence-free"分量）。

但训练我们只需**一个**合理的 $u_t$。

---

## §3 FM Loss（不可直接计算）

### 3.1 朴素 loss

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim U[0,1], x \sim p_t}\left[\|v_\theta(x, t) - u_t(x)\|^2\right]$$

### 3.2 为什么不可行

我们**不知道** $u_t(x)$ 的解析形式！

- $p_t$ 是边缘分布，含数据分布信息
- $u_t$ 是从 $p_t$ 推出的（满足 continuity equation）
- 没有 closed form

---

## §4 引入条件路径

### 4.1 关键概念

引入条件随机变量 $x_1 \sim p_{\text{data}}$。定义**条件概率路径** $p_t(x | x_1)$：
- $t=0$：某固定分布（如 $\mathcal{N}(0, I)$）
- $t=1$：在 $x_1$ 附近的点状分布

显然：
$$p_t(x) = \int p_t(x|x_1) p_{\text{data}}(x_1) \, dx_1$$

### 4.2 条件向量场

类似定义条件向量场 $u_t(x | x_1)$，满足条件 continuity equation：
$$\frac{\partial p_t(x|x_1)}{\partial t} + \nabla_x \cdot \bigl(p_t(x|x_1) u_t(x|x_1)\bigr) = 0$$

---

### 4.3 关键性质：边际化

**Lemma 1**：边缘向量场可以表达为：
$$u_t(x) = \int u_t(x|x_1) \cdot \frac{p_t(x|x_1) p_{\text{data}}(x_1)}{p_t(x)} \, dx_1$$

或者用条件期望写：
$$u_t(x) = \mathbb{E}_{x_1 \sim p(x_1|x_t=x)}[u_t(x|x_1)]$$

—— 边缘向量场是条件向量场的（数据后验下的）期望。

**证明**：对 $p_t(x) = \int p_t(x|x_1) p_{\text{data}}(x_1) dx_1$ 两边对 $t$ 求导：
$$\frac{\partial p_t}{\partial t} = \int \frac{\partial p_t(x|x_1)}{\partial t} p_{\text{data}}(x_1) dx_1$$

代入条件 continuity equation：
$$= -\int \nabla_x \cdot (p_t(x|x_1) u_t(x|x_1)) p_{\text{data}}(x_1) dx_1$$

$$= -\nabla_x \cdot \int p_t(x|x_1) u_t(x|x_1) p_{\text{data}}(x_1) dx_1$$

把这与边缘 continuity equation 比较：
$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (p_t u_t)$$

得：
$$p_t(x) u_t(x) = \int p_t(x|x_1) u_t(x|x_1) p_{\text{data}}(x_1) dx_1$$

除以 $p_t(x)$：
$$u_t(x) = \int u_t(x|x_1) \cdot \frac{p_t(x|x_1) p_{\text{data}}(x_1)}{p_t(x)} dx_1$$

注意 $\frac{p_t(x|x_1) p_{\text{data}}(x_1)}{p_t(x)} = p(x_1 | x_t = x)$（Bayes），所以：
$$u_t(x) = \mathbb{E}_{x_1 \sim p(x_1|x)}[u_t(x|x_1)] \quad \square$$

---

## §5 CFM Loss

### 5.1 定义

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_1, x_t \sim p_t(\cdot|x_1)}\left[\|v_\theta(x_t, t) - u_t(x_t | x_1)\|^2\right]$$

**注意**：$x_1 \sim p_{\text{data}}$，然后 $x_t \sim p_t(\cdot | x_1)$。

### 5.2 可计算性

- $u_t(x | x_1)$：我们**设计** $p_t(x|x_1)$ 时让它解析（见 §6）
- $x_t \sim p_t(\cdot|x_1)$：从条件路径采样

所以 $\mathcal{L}_{\text{CFM}}$ 完全可计算。

---

## §6 等价性主定理

### 6.1 Theorem (Lipman et al. 2023)

对所有 $\theta$：
$$\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)$$

即：训练 CFM 等价于训练 FM（关于参数的梯度相同）。

---

### 6.2 证明

展开 CFM 平方项：
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}\left[\|v_\theta\|^2 - 2 \langle v_\theta, u_t(x|x_1)\rangle + \|u_t(x|x_1)\|^2\right]$$

第一项 $\|v_\theta\|^2$ 不依赖 $x_1$（在 $\theta$ 与 $x_t$ 给定时），所以：
$$\mathbb{E}_{x_1, x_t}[\|v_\theta(x_t, t)\|^2] = \mathbb{E}_{x_t}[\|v_\theta(x_t, t)\|^2]$$

第三项 $\|u_t(x|x_1)\|^2$ 与 $\theta$ 无关，求梯度为 0。

第二项是关键：
$$\mathbb{E}_{x_1, x_t}[\langle v_\theta(x_t, t), u_t(x_t | x_1)\rangle]$$

把 $x_1$ 的期望"换"到内部：
$$= \mathbb{E}_{x_t}\left[\langle v_\theta(x_t, t), \mathbb{E}_{x_1|x_t}[u_t(x_t|x_1)]\rangle\right]$$
$$= \mathbb{E}_{x_t}\left[\langle v_\theta(x_t, t), u_t(x_t)\rangle\right] \quad \text{(用 Lemma 1)}$$

类似展开 FM loss：
$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_t}[\|v_\theta\|^2 - 2\langle v_\theta, u_t(x_t)\rangle + \|u_t\|^2]$$

两个 loss 的差异：
$$\mathcal{L}_{\text{CFM}} - \mathcal{L}_{\text{FM}} = \mathbb{E}[\|u_t(x|x_1)\|^2 - \|u_t(x)\|^2]$$

—— 这个差是**与 $\theta$ 无关的常数**（关于训练）。

所以梯度相同：
$$\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}} \quad \square$$

---

### 6.3 启示

虽然 $\mathcal{L}_{\text{CFM}}$ 和 $\mathcal{L}_{\text{FM}}$ 数值上**不同**，但它们对 $\theta$ 的偏导**完全相同**。

所以：**用 CFM 训练，等于（梯度意义上）训练 FM**。

---

## §7 设计条件路径与向量场

### 7.1 通用形式

设条件路径形如：
$$p_t(x|x_1) = \mathcal{N}(\mu_t(x_1), \sigma_t^2 I)$$

其中 $\mu_t(x_1), \sigma_t$ 是设计的函数：
- $t=0$：$\mu_0 = 0, \sigma_0 = 1$（标准高斯）
- $t=1$：$\mu_1 = x_1, \sigma_1 \approx 0$（点状）

### 7.2 推导条件向量场

设 $x_t = \mu_t(x_1) + \sigma_t \epsilon$（重参数化，$\epsilon \sim \mathcal{N}(0, I)$）。

那么：
$$\frac{dx_t}{dt} = \dot\mu_t(x_1) + \dot\sigma_t \epsilon$$

由 $\epsilon = (x_t - \mu_t)/\sigma_t$：
$$\dot x_t = \dot\mu_t + \dot\sigma_t \cdot \frac{x_t - \mu_t}{\sigma_t}$$

这就是 $u_t(x_t | x_1)$。

---

### 7.3 Linear Path（Rectified Flow）

取 $\mu_t = t \cdot x_1, \sigma_t = 1 - t$：
- $t=0$：$\mu=0, \sigma=1$ ✓
- $t=1$：$\mu=x_1, \sigma=0$ ✓

$x_t = t \cdot x_1 + (1-t) \cdot \epsilon$

$\dot\mu_t = x_1, \dot\sigma_t = -1$：

$$u_t(x_t | x_1) = x_1 - \epsilon = x_1 - \frac{x_t - t x_1}{1-t} \cdot (\text{caveat})$$

等等，让我们更仔细。从 $x_t = t x_1 + (1-t) \epsilon$ 直接：
$$\frac{dx_t}{dt} = x_1 - \epsilon$$

注意：这是**沿 $\epsilon$ 固定的方向**的全导数。所以：
$$u_t(x_t | x_1, \epsilon) = x_1 - \epsilon$$

**简洁极了**——条件向量场就是数据与噪声之差！

---

### 7.4 训练 loss（具体）

$$\mathcal{L}_{\text{Rect}} = \mathbb{E}_{t, x_1, \epsilon}\left[\|v_\theta(x_t, t) - (x_1 - \epsilon)\|^2\right]$$

其中 $x_t = t x_1 + (1-t) \epsilon$。

—— 这就是 SD 3 / Pi-0 用的 loss。

---

## §8 OT 视角

### 8.1 Optimal Transport 路径

OT 给出从 $p_0$ 到 $p_1$ 的**最优** map（最短 Wasserstein 距离）。

对 $p_0 = \mathcal{N}(0, I)$ 和经典 OT 假设下，最优路径在期望上是 linear 的：
$$x_t = (1-t) x_0 + t x_1$$

这正是 rectified flow 的形式（粒子级别的解释）。

---

### 8.2 为什么 linear 好

**几何**：直线是 OT path → ODE 数值积分的"误差"最小（直线不弯）。
**实验**：Liu et al. (2022) 证实 linear path 在少步采样下 FID 优于 DDPM。

---

## §9 Reflow（拉直 trick）

### 9.1 动机

虽然 OT 给的 path 是 linear，但**模型学到的** $v_\theta$ 在采样过程中产生的 trajectory **未必是直线**（因为多个 $x_1$ 可能对应同一 $x_0$，让模型在中间产生 detour）。

### 9.2 Reflow 算法

1. 用第一阶段训好的 $v_\theta^{(1)}$ 采样得到 $(x_0, x_1^{\text{pred}})$ pairs（成千上万）
2. 把 $x_1 = x_1^{\text{pred}}$ 而非真实数据，重新训 $v_\theta^{(2)}$
3. 现在 $(x_0, x_1)$ pairs 是"模型自己的轨迹端点"——拉直了

实证：reflow 后 1-step sampling FID 显著改善。

---

## §10 Generalized FM（拓展）

### 10.1 Stochastic Interpolants (Albergo 2023)

更一般的形式：
$$x_t = \alpha_t \cdot x_1 + \sigma_t \cdot \epsilon + \beta_t \cdot \xi$$

加入 $\xi$ 是另一个随机源。可以 unify DDPM 和 FM。

### 10.2 Schrodinger Bridge

OT 加上"路径熵"约束。给出概率最大的 stochastic path（不是 deterministic）。

详见 De Bortoli 2021。

---

## §11 自查题

1. 复述 CFM 与 FM 等价性证明（不看讲义）
2. 推导 linear path 的 $u_t(x|x_1)$
3. 解释为什么 reflow 能改善 1-step quality
4. 写出 OT 路径与 linear path 的关系

---

## §12 参考

- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023
- Liu et al., *Rectified Flow*, ICLR 2023
- Albergo & Vanden-Eijnden, *Stochastic Interpolants*, 2023
- Esser et al., *SD 3*, 2024
