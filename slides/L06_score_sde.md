# Lecture 06｜Score SDE：DDPM 与 Score Matching 的统一视角

> **本讲目标**
> - 理解 SDE / ODE 视角下的扩散模型
> - 看到 DDPM 与 NCSN 如何在同一框架下统一（VP-SDE vs VE-SDE）
> - 掌握 reverse-time SDE 公式（Anderson 1982）
> - 理解 probability flow ODE 与确定性采样
> - 为 W6（DDIM）打下数学基础

> **对应论文**：Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021

> **难度警告**：本讲是整个课程的"数学高峰"。建议预留 4 小时阅读 + 配合推导手稿 derive_04 学习。
>
> 如果对 ODE/SDE 数值方法、连续极限或 probability flow ODE 的直觉不稳，可以先看 [Score SDE 数学基础建议顺序](../supplementary/README.md#score-sde-数学基础建议顺序)。

---

## §1 动机：为什么需要连续时间视角？

L03-L05 我们用的是离散时间扩散：$x_0, x_1, \dots, x_T$，$T=1000$。

**问题**：
- $T$ 是个 magic number，为什么是 1000 而不是 100 或 10000？
- 离散步骤之间的"间隙"如何理解？
- DDPM（噪声预测）和 NCSN（score 估计）看起来非常像，但数学关系不清晰

**Score SDE 的洞察**：让 $T \to \infty$，$\Delta t \to 0$，整个过程变成连续 SDE。
- 离散步数只是数值离散化的细节
- DDPM 和 NCSN 是同一连续 SDE 的两种不同离散化

---

## §2 SDE 基础速通

### 2.1 ODE vs SDE

**ODE**（常微分方程）：$\frac{dx}{dt} = f(x, t)$ — 确定性轨迹

**SDE**（随机微分方程）：
$$dx = f(x, t) \, dt + g(t) \, dW$$

- $f(x, t)$：**drift**（确定性"漂移"方向）
- $g(t)$：**diffusion coefficient**（噪声强度）
- $dW$：Wiener 过程的"无穷小增量"

**数值近似**（Euler-Maruyama）：
$$x_{t+\Delta t} = x_t + f(x_t, t) \cdot \Delta t + g(t) \cdot \sqrt{\Delta t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

---

### 2.2 Fokker-Planck 方程

给定 SDE $dx = f \, dt + g \, dW$，对应的密度 $p_t(x)$ 演化遵循 Fokker-Planck 方程：

$$\frac{\partial p_t}{\partial t} = -\nabla_x \cdot (f \cdot p_t) + \frac{1}{2} g^2 \nabla_x^2 p_t$$

**意义**：SDE 描述粒子轨迹，Fokker-Planck 描述粒子密度的演化。两者等价。

> **🔑 在 Score SDE 中**：我们关心的是密度演化（生成模型本质），但要采样必须有 SDE 形式。两套语言互译。

---

## §3 把 DDPM 写成 SDE

### 3.1 离散过程的极限

DDPM 单步：
$$x_{t} = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon$$

设 $\beta_t = \beta(t) \Delta t$（密度型参数化），$\Delta t = 1/T$，取极限 $T \to \infty$：

**Taylor 展开** $\sqrt{1 - \beta(t)\Delta t} \approx 1 - \frac{1}{2} \beta(t) \Delta t$：

$$x_{t} \approx x_{t-1} - \frac{1}{2} \beta(t) \Delta t \cdot x_{t-1} + \sqrt{\beta(t) \Delta t} \cdot \epsilon$$

对应的 SDE：

$$\boxed{dx = -\frac{1}{2} \beta(t) x \, dt + \sqrt{\beta(t)} \, dW}$$

这叫 **Variance Preserving SDE (VP-SDE)**。

---

### 3.2 NCSN 写成 SDE

回忆 NCSN（W1）：用一系列噪声尺度 $\sigma_1 > \sigma_2 > \dots$ 训练 score 网络。

把噪声尺度也连续化：$\sigma(t)$。则加噪过程是：
$$x_t = x_0 + \sigma(t) \cdot \epsilon$$

对应的 SDE：

$$\boxed{dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, dW}$$

这叫 **Variance Exploding SDE (VE-SDE)**——方差随时间无界增长。

---

### 3.3 两类 SDE 的对比

| 性质 | VP-SDE（DDPM） | VE-SDE（NCSN） |
|------|----------------|----------------|
| Drift | $-\frac{1}{2}\beta(t) x$（向原点收缩） | 0 |
| Diffusion | $\sqrt{\beta(t)}$ | $\sqrt{d\sigma^2/dt}$ |
| $\text{Var}(x_T)$ | 有界（→ 1） | 无界（→ ∞） |
| 终态分布 | $\mathcal{N}(0, I)$ | $\mathcal{N}(0, \sigma_T^2 I)$，$\sigma_T$ 极大 |

**意义**：DDPM 和 NCSN 不是两种独立方法，而是**同一框架下选择不同的 $f, g$**。

---

## §4 反向 SDE：Anderson 公式

### 4.1 神奇的事实

**定理（Anderson 1982）**：对任意 forward SDE $dx = f \, dt + g \, dW$，存在一个对应的**反向 SDE**，描述如何从终态采样回初态：

$$\boxed{dx = \left[ f(x, t) - g(t)^2 \cdot \nabla_x \log p_t(x) \right] dt + g(t) \, d\bar W}$$

其中 $d\bar W$ 是从 $T$ 到 $0$ 反向的 Wiener 过程，$\nabla_x \log p_t(x)$ 是 $t$ 时刻边缘密度的 score。

---

### 4.2 解读

**反向 drift**：
- 正向的 $f(x, t)$（仍然存在）
- 减去 $g(t)^2 \cdot s_t(x)$：朝高密度方向"反推"

**关键观察**：要做反向采样，**只需要知道 score $\nabla_x \log p_t(x)$**。其他都是 forward SDE 决定的。

> **🔑 这就是 Score SDE 的核心洞察**：所有扩散模型最终都归结到"如何估计每个时间点的 score"。

---

### 4.3 训练目标

用 score matching 训练神经网络 $s_\theta(x, t)$ 逼近真实 score：

$$\mathcal{L} = \mathbb{E}_t \left\{ \lambda(t) \mathbb{E}_{x_0} \mathbb{E}_{x_t | x_0} \left[ \| s_\theta(x_t, t) - \nabla_{x_t} \log p(x_t | x_0) \|^2 \right] \right\}$$

由于 $p(x_t | x_0)$ 是闭合高斯（VP-SDE 中等价于 DDPM 的 $q(x_t|x_0)$），$\nabla_{x_t} \log p(x_t|x_0)$ 可解析：

$$\nabla_{x_t} \log p(x_t | x_0) = -\frac{x_t - \mu_t}{\sigma_t^2}$$

代入 VP-SDE 中 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$：

$$\nabla_{x_t} \log p(x_t | x_0) = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

—— **预测 score 与预测 $\epsilon$ 差一个时间相关系数**！这是 DDPM 与 score-based 完全等价的严格证明。

---

## §5 Probability Flow ODE：去除随机性

### 5.1 神奇的事实之二

对任意 SDE $dx = f \, dt + g \, dW$，存在一个**确定性 ODE**：

$$\boxed{\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \cdot \nabla_x \log p_t(x)}$$

它在每个时间点 $t$ 的**边缘分布 $p_t(x)$ 与原 SDE 完全相同**，但轨迹是**确定性的**。

这叫 **probability flow ODE**。

> 补充阅读：这句话最容易卡在“边缘分布相同但轨迹不同”，可参考 [Probability Flow ODE：如何理解“边缘分布完全相同”](../supplementary/probability-flow-ode-marginals.md)。

---

### 5.2 为什么这个 ODE 重要？

**优势**：
1. **可逆**：从 $x_T$ 倒着积分到 $x_0$ 再正着积分回 $x_T$ 应当回到同一点
2. **可计算精确 likelihood**：通过 instantaneous change of variables 公式
3. **可用高阶 ODE 求解器**（Runge-Kutta、DPM-Solver）加速采样
4. **去随机性**：生成结果可复现（给定初始噪声，输出固定）

> **🔑 这是 DDIM 的数学基础**：DDIM（W6）本质上就是 probability flow ODE 的一阶离散化。
>
> 这四点的展开解释见 [Probability Flow ODE 补充阅读 §8](../supplementary/probability-flow-ode-marginals.md#8-为什么这个-ode-重要)。

---

### 5.3 ODE 视角下的扩散模型

回顾本课程的工具进展：

```
W2-W4: 离散时间，固定步数（DDPM/Improved DDPM）
              ↓
W5:    连续时间，统一框架（Score SDE）
              ↓
W6:    转化为 ODE，高阶求解器加速
              ↓
W11:   Flow Matching（直接学 ODE 向量场，不经过 SDE）
              ↓
W12:   Consistency Models（学一步映射）
```

每一步都是"扔掉某些不必要的东西"，让推理越来越高效。

---

## §6 Predictor-Corrector 采样

### 6.1 单纯 SDE 求解的问题

数值离散化的 SDE 求解器（如 Euler-Maruyama）有截断误差，每步引入一些偏差。

**Predictor-Corrector** 思路：
- **Predictor**：用 SDE 求解器走一步（如 reverse SDE Euler-Maruyama）
- **Corrector**：用 Langevin MCMC 在当前 $t$ 校正回正确分布

```
for t = T, T-Δt, ..., 0:
    # Predictor: 一步 reverse SDE
    x = x + [f(x, t) - g²(t) · s_θ(x, t)] · (-Δt) + g(t) · √Δt · z

    # Corrector: K 步 Langevin
    for k = 1, ..., K:
        x = x + ε · s_θ(x, t) + √(2ε) · z'
```

**效果**：FID 在 CIFAR-10 上从 3.2 降到 2.4。这是 Score SDE 论文的最强结果。

---

## §7 数值算法对比

下表汇总了 Score SDE 论文中的几种采样器：

| 算法 | 类型 | 步数 | CIFAR-10 FID |
|------|------|------|--------------|
| Ancestral (DDPM) | SDE 离散 | 1000 | 3.17 |
| Euler-Maruyama | SDE 求解 | 1000 | 4.45 |
| Reverse Diffusion + Langevin (PC) | Predictor-Corrector | 1000+1000 | **2.41** |
| Probability Flow ODE (RK45) | ODE 求解 | 自适应 | 2.87 |

**关键观察**：
- ODE 与 SDE 给出可比质量
- PC 采样器实现 SOTA（但代价是步数翻倍）

---

## §8 一个统一图景

```
                      ┌──────────────────┐
                      │   Score SDE 框架  │
                      └────────┬─────────┘
                               │
                ┌──────────────┴──────────────┐
                │                              │
       ┌────────▼─────────┐          ┌────────▼─────────┐
       │   VP-SDE         │          │   VE-SDE         │
       │   (DDPM 极限)    │          │   (NCSN 极限)    │
       └────────┬─────────┘          └────────┬─────────┘
                │                              │
       ┌────────▼─────────┐          ┌────────▼─────────┐
       │  离散化 → DDPM   │          │  离散化 → NCSN   │
       │  Anc sampling    │          │  Annealed Langevin│
       └──────────────────┘          └──────────────────┘
                │                              │
                └──────────────┬───────────────┘
                               │
                ┌──────────────▼──────────────┐
                │   Probability Flow ODE        │
                │   → DDIM, DPM-Solver, EDM    │
                └─────────────────────────────┘
```

每个箭头都是某种 reduction 或近似。理解这张图后，看任何新的"diffusion sampler"论文都能立刻定位。

---

## §9 本讲核心要点

1. **连续时间视角**让我们超越 $T=1000$ 的离散细节
2. **DDPM 是 VP-SDE 的特殊离散化，NCSN 是 VE-SDE 的特殊离散化**
3. **Anderson 公式**：反向 SDE 只需 score 即可
4. **Probability Flow ODE**：确定性的、可逆的、可计算 likelihood 的等价过程
5. **预测 $\epsilon$ ↔ 预测 score** 在数学上严格等价
6. **Predictor-Corrector** 是 SDE 视角下的 SOTA 采样器

---

## §10 课后任务

### 必做

1. **手推 Anderson 公式**（中间步骤可参考 derive_04）：
   - 用 Fokker-Planck 方程出发
   - 推出 reverse-time SDE 的 drift 公式
   - 标注每步使用的工具（链式法则、积分分部）

2. **代码任务**：用 PyTorch 实现 2D 玩具数据集（混合高斯）上的：
   - VP-SDE 加噪可视化
   - Score matching 训练
   - Reverse SDE 采样 + probability flow ODE 采样对比
   提交 `score_sde_toy.ipynb`。

3. **Reading note**：阅读 Song et al. 2021 论文 §1-§3 + §5。重点理解：
   - VP/VE SDE 的形式
   - Anderson 公式的物理意义
   - PC sampler 的两阶段含义

### 选做

4. **概念题**：DDPM 中 $\beta_t$ 的"linear schedule"对应 VP-SDE 中 $\beta(t) = ?$ 的连续函数？
   写下推导。

5. 阅读 Karras 2022 EDM 论文前 4 节，理解他们如何把 Score SDE 推广到任意 $f, g$ 选择。

---

## §11 推荐进一步阅读

| 资源 | 重点 |
|------|------|
| Score SDE 原论文（Song 2021） | 必读 |
| Karras 2022 *Elucidating Design Space* | Score SDE 的简化重写，更易实现 |
| Yang Song 博客 *Score-Based Models* | 直觉理解 |
| 推导手稿 `derive_04_score_sde.pdf` | Anderson 公式完整推导 |
| [ODE 与 SDE 数值方法入门](../supplementary/ode-sde-numerical-methods.md) | Taylor、Euler、Euler-Maruyama 与误差阶 |
| [Probability Flow ODE 补充阅读](../supplementary/probability-flow-ode-marginals.md) | 边缘分布相同、可逆、likelihood 与 DDIM 关系 |
| Lu 2022 *DPM-Solver* §2 | 工程化视角下的 ODE 求解 |

---

## §12 常见问题

**Q: 我看 Score SDE 论文觉得数学很难，能不能跳过直接学 DDIM？**

A: 短期可以，长期不行。DDIM 看似简单，但若不理解它是 probability flow ODE 的一阶近似，你看到 DPM-Solver、DEIS 等后续 sampler 时会完全迷失。本讲的核心不是细节推导，而是**建立"SDE-ODE 二元视角"的直觉**。

**Q: 为什么 NCSN 和 DDPM 看起来训练目标几乎一样，却分属不同范式？**

A: 训练目标确实数学等价（差一个时间系数）。但工程上：
- DDPM 用 VP-SDE 离散化（保持方差 ≈ 1）
- NCSN 用 VE-SDE 离散化（方差随 $\sigma$ 增长）

两种归一化方式让网络看到的输入分布不同，因此网络架构、调参经验都不同。Score SDE 之后大家逐渐统一到 VP 路线（Stable Diffusion 等）。

**Q: Probability flow ODE 真的"确定性"？那为什么不一直用它？**

A: ODE 在采样时确定，**但训练阶段仍然依赖 SDE 视角**（需要 forward 加噪过程提供训练样本）。"确定性"指的是 inference 时给定初始 $x_T$ 输出唯一。这在某些应用中是优势（可复现），但在其他场合是劣势（缺少 sample diversity）。

**Q: 我能不能不学 SDE 视角，只学 DDIM？**

A: 可以学得动 DDIM，但你会觉得"这个公式是怎么想出来的？"。学了 Score SDE 后，DDIM 公式会变成"显然就该这样"——这是更深层的理解。

---

> **下一讲预告**：L07 我们进入实战部分——**DDIM 与 DPM-Solver**。前者把采样从 1000 步降到 50 步，后者进一步降到 10 步。本讲建立的 ODE 视角是它们的数学母版。
