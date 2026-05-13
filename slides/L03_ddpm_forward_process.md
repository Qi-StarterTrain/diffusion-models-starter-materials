# Lecture 03｜DDPM 上：Forward Process 与闭合形式

> **本讲目标**
> - 完整理解 DDPM 的前向过程（forward process）的设计动机
> - 推导出 $q(x_t|x_0)$ 的闭合形式（关键公式之一）
> - 理解 noise schedule（$\beta_t$）的设计哲学
> - 推导 $q(x_{t-1} | x_t, x_0)$ 的闭合形式（反向训练目标的母版）
> - 建立"为什么这样设计 forward process"的直觉

---

## §1 DDPM 的总体框架

### 1.1 三个核心问题

理解 DDPM 需要回答三个问题：

1. **如何把数据变成噪声？**（forward process）
2. **如何从噪声变回数据？**（reverse process）
3. **如何训练这个反向网络？**（loss function）

本讲解决问题 1，下一讲（L04）解决问题 2 和 3。

---

### 1.2 DDPM 的图景

```
Forward (固定，不学习):
  x_0  →  x_1  →  x_2  →  ...  →  x_T
  数据                              ≈ 纯高斯噪声
  ──────加噪逐步进行──────────────→

Reverse (学习):
  x_T  →  x_{T-1}  →  ...  →  x_1  →  x_0
  噪声                                 数据
  ←──────去噪逐步进行────────────────
```

- Forward：**固定**的高斯加噪，定义为 $q(x_t | x_{t-1})$
- Reverse：**学习**的高斯去噪，参数化为 $p_\theta(x_{t-1} | x_t)$
- 训练目标：让 $p_\theta$ 学着"反过来走"

这个设计巧妙之处在于——**只要 forward 过程定义得好，reverse 过程就有清晰的训练目标**。

---

## §2 Forward Process 的定义

### 2.1 单步加噪

DDPM 把每一步加噪定义为一个高斯转移：

$$q(x_t | x_{t-1}) = \mathcal{N}\bigl(x_t; \sqrt{1 - \beta_t} \cdot x_{t-1}, \beta_t \cdot I\bigr)$$

其中 $\beta_t \in (0, 1)$ 是预先指定的"噪声强度"序列（noise schedule）。

**用重参数化展开**：
$$x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

---

### 2.2 这个定义的精妙之处

为什么是 $\sqrt{1-\beta_t}$ 系数而不是简单的 $x_t = x_{t-1} + \sqrt{\beta_t}\epsilon$？

**关键性质**：上述定义**保持了方差稳定**。

设 $\mathrm{Var}(x_{t-1}) = 1$（数据归一化后），则：
$$\mathrm{Var}(x_t) = (1-\beta_t) \cdot 1 + \beta_t \cdot 1 = 1$$

—— **方差不会随时间步爆炸**！如果用简单加法，方差会单调累积，最终无法控制。

> **🔑 设计哲学**：每一步既"丢失一点信号"（$\sqrt{1-\beta_t}$ 系数），又"加入一点噪声"（$\sqrt{\beta_t}$ 系数），两者比例使总方差守恒。

这种性质叫做 **variance preserving (VP)**。

---

### 2.3 完整 forward 过程

整个 forward 过程是马尔可夫链：

$$q(x_{1:T} | x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})$$

—— 每一步只依赖前一步。

---

## §3 关键推导：$q(x_t | x_0)$ 的闭合形式

### 3.1 为什么需要这个公式？

直接采样 $x_t$ 需要从 $x_0$ 一步步采到 $x_t$，要做 $t$ 次操作。如果有闭合形式，**可以一步到位**——这对训练效率至关重要（每个 batch 中要随机采样不同的 $t$）。

---

### 3.2 引入符号简化

定义：
$$\alpha_t := 1 - \beta_t, \quad \bar\alpha_t := \prod_{s=1}^{t} \alpha_s$$

—— $\bar\alpha_t$ 是从 1 到 $t$ 所有 $\alpha$ 的累积乘积。

---

### 3.3 推导（**必须完全掌握**）

我们要证明：
$$q(x_t | x_0) = \mathcal{N}\bigl(x_t; \sqrt{\bar\alpha_t} \cdot x_0, (1 - \bar\alpha_t) \cdot I\bigr)$$

**用归纳法**。

**第 1 步（base case）**：
$$x_1 = \sqrt{\alpha_1} \cdot x_0 + \sqrt{1 - \alpha_1} \cdot \epsilon_1, \quad \epsilon_1 \sim \mathcal{N}(0, I)$$

故 $q(x_1 | x_0) = \mathcal{N}(\sqrt{\alpha_1} x_0, (1-\alpha_1) I) = \mathcal{N}(\sqrt{\bar\alpha_1} x_0, (1-\bar\alpha_1) I)$。✓

**第 2 步（归纳步）**：假设 $x_{t-1} = \sqrt{\bar\alpha_{t-1}} x_0 + \sqrt{1 - \bar\alpha_{t-1}} \cdot \tilde\epsilon$，$\tilde\epsilon \sim \mathcal{N}(0, I)$。

代入单步公式：
$$
\begin{aligned}
x_t &= \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{1-\alpha_t} \cdot \epsilon_t \\
&= \sqrt{\alpha_t} \left( \sqrt{\bar\alpha_{t-1}} x_0 + \sqrt{1-\bar\alpha_{t-1}} \tilde\epsilon \right) + \sqrt{1-\alpha_t} \cdot \epsilon_t \\
&= \sqrt{\alpha_t \bar\alpha_{t-1}} \cdot x_0 + \sqrt{\alpha_t (1-\bar\alpha_{t-1})} \cdot \tilde\epsilon + \sqrt{1-\alpha_t} \cdot \epsilon_t \\
&= \sqrt{\bar\alpha_t} \cdot x_0 + \underbrace{\sqrt{\alpha_t (1-\bar\alpha_{t-1})} \cdot \tilde\epsilon + \sqrt{1-\alpha_t} \cdot \epsilon_t}_{\text{两个独立高斯之和}}
\end{aligned}
$$

利用"独立高斯之和的方差相加"：
$$\mathrm{Var} = \alpha_t (1-\bar\alpha_{t-1}) + (1-\alpha_t) = \alpha_t - \alpha_t \bar\alpha_{t-1} + 1 - \alpha_t = 1 - \bar\alpha_t$$

所以这两项之和等价于一个 $\sqrt{1-\bar\alpha_t} \cdot \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$。

最终：
$$\boxed{x_t = \sqrt{\bar\alpha_t} \cdot x_0 + \sqrt{1-\bar\alpha_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)}$$

即：
$$\boxed{q(x_t | x_0) = \mathcal{N}\bigl(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I\bigr)}$$

✅ 这是 DDPM 中**最常用的公式**，必须能闭眼写出。

---

### 3.4 直观解释

公式 $x_t = \sqrt{\bar\alpha_t} \cdot x_0 + \sqrt{1-\bar\alpha_t} \cdot \epsilon$ 的两项分别是：

- $\sqrt{\bar\alpha_t} \cdot x_0$：原始信号被衰减 $\sqrt{\bar\alpha_t}$ 倍
- $\sqrt{1-\bar\alpha_t} \cdot \epsilon$：加入幅度为 $\sqrt{1-\bar\alpha_t}$ 的高斯噪声

随 $t$ 增大：
- $\bar\alpha_t \to 0$：信号占比 $\to 0$
- $1 - \bar\alpha_t \to 1$：噪声占比 $\to 1$

最终 $x_T$ 接近纯噪声 $\mathcal{N}(0, I)$。

---

### 3.5 信噪比（SNR）视角

定义：
$$\text{SNR}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t}$$

—— 信号方差 / 噪声方差。

**关键事实**：DDPM 的所有训练动力学（loss 权重、采样策略、condition 信息）都可以用 SNR 来重新理解。后续 EDM、Karras 2022 等工作把 SNR 视为更基础的设计变量。

---

## §4 Noise Schedule 设计

### 4.1 Linear Schedule（DDPM 原始）

$$\beta_t = \beta_{\min} + \frac{t-1}{T-1} (\beta_{\max} - \beta_{\min})$$

DDPM 默认：$T=1000$，$\beta_{\min}=10^{-4}$，$\beta_{\max}=0.02$。

---

### 4.2 问题：Linear Schedule 有什么不好？

下图（概念）：在 linear schedule 下，$\bar\alpha_t$ 在前期下降很快。

```
β_t (linear):     ────────/──────/──────  随 t 线性增长
ᾱ_t (linear):     ‾─────────────\_______ 前期快速下降
```

**结果**：低 $t$ 时图像就已经被破坏得很厉害，很多时间步浪费在"已经基本是噪声"的状态。

---

### 4.3 Cosine Schedule（Improved DDPM）

Nichol & Dhariwal 2021 提出：
$$\bar\alpha_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \right)$$

其中 $s = 0.008$ 防止 $t=0$ 时数值问题。

**优势**：
- 低 $t$ 区域 $\bar\alpha_t$ 下降更平缓，给模型更多"细微去噪"的训练信号
- 高 $t$ 区域 $\bar\alpha_t$ 接近 0，保证 $x_T$ 足够接近纯噪声

```
ᾱ_t (linear):  ‾────\─────\\\\─────────  前期陡降
ᾱ_t (cosine):  ‾────────\\─────\─────___ 平滑过渡
```

> **🔑 经验**：在 64×64 等低分辨率上，cosine 比 linear 提升 1–3 个 FID 点。

---

### 4.4 其他 schedule

- **Sigmoid schedule**（最近常用）：在两端更平缓
- **Karras schedule**（EDM, 2022）：基于 SNR 重新参数化，对采样质量有显著提升
- **Continuous schedule**：$T \to \infty$ 极限下的 SDE 视角

---

## §5 关键推导：$q(x_{t-1} | x_t, x_0)$ 的闭合形式

### 5.1 为什么这个分布重要？

我们最终要训练一个网络拟合 $p_\theta(x_{t-1} | x_t)$。直接拟合无法处理（不知道真实分布是什么）。

但如果**已知 $x_0$**，我们可以解析地写出 $q(x_{t-1} | x_t, x_0)$——这就是反向过程的"理想答案"。

DDPM 的训练目标就是让 $p_\theta(x_{t-1} | x_t)$ 逼近这个理想分布。

---

### 5.2 推导

由贝叶斯定理：
$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

由马尔可夫性，$q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$。三项都是已知的高斯：

- $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, \beta_t I)$
- $q(x_{t-1} | x_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}} x_0, (1-\bar\alpha_{t-1}) I)$
- $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$

**经过繁琐但机械的高斯运算**（详见推导手稿 derive_02），得到：

$$\boxed{q(x_{t-1} | x_t, x_0) = \mathcal{N}\bigl(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t \cdot I\bigr)}$$

其中：
$$\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} x_t$$

$$\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$$

—— 是 $x_t$ 和 $x_0$ 的线性组合的高斯，**方差不依赖于 $x_t$ 或 $x_0$**！

---

### 5.3 改写成 noise prediction 形式

利用 $x_0 = \frac{1}{\sqrt{\bar\alpha_t}}(x_t - \sqrt{1-\bar\alpha_t} \epsilon)$，代入 $\tilde\mu_t$ 并化简：

$$\tilde\mu_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon \right)$$

—— **均值是 $x_t$ 减去一个跟 $\epsilon$ 成比例的项**！

> **🔑 启示**：如果我们让网络预测 $\epsilon$，则反向均值就是确定的——这指引了 DDPM 的训练目标设计（下一讲详细展开）。

---

## §6 直观可视化（必看）

下图是一张 CIFAR-10 图像在不同 $t$ 下加噪后的样子：

```
t=0     t=100   t=200   t=400   t=600   t=800   t=1000
[原图]  [清晰]   [模糊]   [扭曲]   [接近噪声]    [纯噪声]
```

观察规律：
1. 前 100–200 步，图像还很清晰，说明 forward 过程在低 $t$ 区域只在做细节扰动
2. 中间 300–600 步，图像主要内容仍可辨认但变得模糊
3. 后 700+ 步，图像迅速变成噪声

**这正是 cosine schedule 试图改进的部分**：linear schedule 在 t=200 已经破坏太多，cosine 推迟了破坏速度。

---

## §7 本讲核心要点回顾

1. Forward process 是**固定**的高斯加噪，每步定义为 $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, \beta_t I)$
2. **方差守恒**性质保证 $x_t$ 不发散
3. **闭合形式 $q(x_t|x_0)$**：$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$（**必背**）
4. Noise schedule 设计影响训练效率：cosine 优于 linear
5. **理想反向后验 $q(x_{t-1}|x_t,x_0)$ 是闭合高斯**，给我们提供训练目标
6. 改写后看到：均值的形式暗示"预测 $\epsilon$ 即可"

---

## §8 课后任务

### 必做

1. **手推 $q(x_t|x_0)$**：完整重做 §3.3 的推导，标注每一步使用的工具，提交手写或 LaTeX 版本。

2. **手推 $q(x_{t-1}|x_t,x_0)$**：参考推导手稿 derive_02，自己重做高斯运算，得到 $\tilde\mu_t$ 和 $\tilde\beta_t$。

3. **代码任务**：实现 forward process 可视化
   ```python
   # 实现 q_sample(x0, t) 函数
   # 选 1 张 CIFAR-10 图像
   # 在 t=[0, 100, 200, ..., 1000] 下分别采样并保存
   # 拼成一张对比图
   ```
   提交 `forward_process_visualization.py` 和生成的对比图。

4. **schedule 对比**：实现 linear 和 cosine 两种 schedule，画出 $\bar\alpha_t$ 随 $t$ 的变化曲线。

### 选做

5. **思考题**：如果 $\beta_t$ 设得太大（如 $\beta_{\max}=0.5$），会发生什么？写一段代码验证你的猜测。

6. 阅读 Karras 2022 *Elucidating the Design Space*（EDM）前 3 节，理解为什么作者认为 SNR 是更基础的参数。

---

## §9 推荐进一步阅读

| 资源 | 重点 |
|------|------|
| DDPM 原始论文 §2-§3 | 标准推导 |
| Lilian Weng *What are Diffusion Models?* | 同一推导的另一种叙述 |
| Improved DDPM 论文 §3 | cosine schedule 设计动机 |
| 推导手稿 `derive_02_ddpm_forward.pdf` | 完整带注释推导 |

---

## §10 常见问题

**Q: 为什么前向过程要定义成马尔可夫链？**

A: 马尔可夫性极大简化了推导（联合分布可以分解、闭合形式可推）。如果 forward 不是马尔可夫，整个 ELBO 推导都会复杂得多。后续 DDIM 论文展示了如何"放宽"马尔可夫性以加速采样。

**Q: $T$ 必须是 1000 吗？**

A: 不是。$T$ 是个超参数。DDPM 选 1000 是因为足够细，让 $q(x_{t-1}|x_t)$ 接近高斯（小步长假设）。在 score SDE 视角下，$T \to \infty$ 时整个过程变成连续 SDE。

**Q: 为什么 forward 过程不学习？**

A: 这是 DDPM 设计的关键决策。固定 forward 让我们能解析地推出 $q(x_t|x_0)$ 和 $q(x_{t-1}|x_t,x_0)$，从而得到清晰的训练目标。如果 forward 也学习（如某些 normalizing flow），就失去了这种数学便利。

**Q: 实际工程中我看到代码里有 `betas = torch.linspace(1e-4, 0.02, 1000)`，但好的论文用 cosine。我应该用哪个？**

A: 跑 baseline 用 linear（与论文对齐），自己做 SOTA 实验用 cosine 或 sigmoid。代码框架应支持轻松切换 schedule（这就是为什么 Project 1 中要把 schedule 抽成单独模块）。

---

> **下一讲预告**：L04 我们将完成 DDPM 三部曲——
> ① 推导 reverse process 的训练目标（ELBO 化简到 MSE）
> ② 看懂 U-Net 架构在 DDPM 中的设计
> ③ 写出完整采样算法
> 这一讲是整个课程的"核中之核"，请务必先完成本讲的手推作业再进入。
