# L12: Flow Matching（W11）

> **目标**：理解 SD 3、Flux、Pi-0 共享的"新范式"
> **前置**：L06 (Score SDE)，L11 (DiT)
> **核心问题**：能不能不学 score、不学 noise，**直接学一个向量场**？

---

## §1 课程定位

Flow Matching (FM) 是 2023 年开始爆火、2024 年成为主流的新范式。

**关键事实**：
- SD 3、Flux、Pi-0 (Physical Intelligence)、AuraFlow 都用 FM
- 数学上比 DDPM 简洁——**没有 SDE，只有 ODE**
- 工程上比 EDM 还容易实现

**核心问题**：为什么扩散模型需要"加噪 → 反向去噪"这么复杂的流程？

---

## §2 直觉：从 OT 出发

### 2.1 最简单的生成问题

我们有：
- 源分布 $p_0$（标准高斯，能采样）
- 目标分布 $p_1$（数据，能采样）

任务：学一个从 $p_0$ 到 $p_1$ 的映射。

---

### 2.2 用 ODE 表示

时间 $t \in [0, 1]$，定义路径 $x_t$ 满足 ODE：
$$\frac{dx}{dt} = v_\theta(x, t)$$

初值 $x_0 \sim p_0$，目标 $x_1 \sim p_1$。

如果学到正确的**向量场** $v(x, t)$，从任意 $x_0$ 走 ODE 到 $t=1$ 就到目标分布。

**问题**：怎么定义"正确的"向量场？

---

### 2.3 与扩散的关系

回顾 probability flow ODE：
$$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla \log p_t(x)$$

这是一个**特殊的**向量场——为了匹配 VP-SDE 的边缘分布而精心设计。

Flow Matching 说：**为什么要绑定到 SDE？直接选一个简洁的向量场！**

---

## §3 概率路径与向量场

### 3.1 概率路径

定义一族**概率分布** $p_t$（$t \in [0, 1]$），其中 $p_0$ 是先验，$p_1$ 是数据。

**路径性质**：$p_t$ 随 $t$ 连续变化。

---

### 3.2 生成向量场

给定路径 $\{p_t\}$，能"生成"它的 ODE 向量场是什么？

满足 **continuity equation**：
$$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \cdot v_t) = 0$$

如果 $v_t$ 满足这方程，ODE $dx/dt = v_t(x)$ 的解 $x_t$ 服从 $p_t$。

---

### 3.3 直接学 $v$？

最朴素的损失：
$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x \sim p_t}\left[\|v_\theta(x, t) - v_t(x)\|^2\right]$$

**但**：$v_t(x)$ 我们不知道（continuity equation 给的是约束，不是闭合形式）。

---

## §4 Conditional Flow Matching（FM 的关键 trick）

### 4.1 条件化思想

定义**条件路径** $p_t(x | x_1)$：从一个具体的数据点 $x_1$ 出发的路径。

**Trick**：
- 条件路径的向量场 $u_t(x | x_1)$ 我们可以**显式定义**
- 边缘化：$p_t(x) = \int p_t(x | x_1) p_{\text{data}}(x_1) dx_1$
- 关键定理（Lipman et al. 2023）：**学习条件向量场等价于学习边缘向量场**

---

### 4.2 关键定理

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_1 \sim p_{\text{data}}, x \sim p_t(\cdot | x_1)} \left[\|v_\theta(x, t) - u_t(x | x_1)\|^2\right]$$

**梯度上等价于** $\mathcal{L}_{\text{FM}}$（关于 $\theta$）。

证明思路（Lipman 论文 §3.2）：
- 展开两个 loss 对 $\theta$ 的梯度
- 用条件期望恒等式 + Tonelli 定理
- 多余的项与 $\theta$ 无关，gradient 一致

---

## §5 最简化：Linear Path（Rectified Flow / SD 3 用）

### 5.1 直线路径

定义条件路径为 $x_0$ 到 $x_1$ 的**直线**：
$$x_t = (1-t) x_0 + t x_1, \quad x_0 \sim \mathcal{N}(0, I), \quad x_1 \sim p_{\text{data}}$$

**对应的条件向量场**：
$$u_t(x_t | x_1) = \frac{d x_t}{dt} = x_1 - x_0$$

—— **就是简单的差向量**！

---

### 5.2 训练 Loss

$$\mathcal{L}_{\text{Rect}} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

**这就是 SD 3 用的训练目标**。

写成代码：
```python
def rectified_flow_loss(model, x_1, t):
    x_0 = torch.randn_like(x_1)
    x_t = (1 - t) * x_0 + t * x_1
    v_pred = model(x_t, t)
    target = x_1 - x_0
    return F.mse_loss(v_pred, target)
```

简洁到爆。**没有 schedule、没有 $\bar\alpha_t$、没有复杂参数化**。

---

### 5.3 推理（ODE 积分）

```python
def sample(model, n_steps=20):
    x = torch.randn(...)  # x_0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = i * dt
        v = model(x, t)
        x = x + v * dt  # Euler
    return x  # 这就是 x_1
```

时间从 0 到 1（**不是 T 到 0**），步长任意。

---

## §6 与扩散模型的关系

### 6.1 数学对比

| | DDPM | Flow Matching (Rectified) |
|---|------|----------------------------|
| 训练目标 | $\|\epsilon - \epsilon_\theta\|^2$ | $\|v_\theta - (x_1-x_0)\|^2$ |
| 时间 | T to 0 | 0 to 1 |
| Schedule | $\beta_t$ 复杂 | 无（线性路径） |
| 反向 | 加噪+去噪 SDE | 直接 ODE |

---

### 6.2 数学等价（重要）

DDPM 的 noise prediction 可以**重写成**一种 flow matching：

对 VP-SDE，定义 $v(x, t) = \dots$（详见 derive_07），可以证明 noise prediction 和 velocity prediction 是同一物的不同参数化。

**关键差别在 schedule**：
- DDPM 用 $\bar\alpha_t$（凸函数）
- Rectified flow 用 $t$（线性）

**实验显示**：linear path 在多步生成时**更直**——直线意味着 ODE 数值积分更精确（少步采样质量好）。

---

### 6.3 实证比较（SD 3 论文）

| Schedule | FID (CFG=2) @ 50 steps |
|---------|------------------------|
| DDPM (Cosine) | 2.85 |
| EDM | 2.45 |
| **Rectified Flow** | **2.23** |

Flow matching 不仅简洁，在大模型规模下还更好。

---

## §7 路径设计

Rectified flow 用直线，但 FM 框架允许**任意路径**。

### 7.1 设计 desideratum

好的路径满足：
1. **闭合**：$p_t(x|x_1)$ 解析可采样
2. **平滑**：$u_t$ 容易学
3. **直线性**：少步积分精确

---

### 7.2 OT 路径

最优传输路径：在 $p_0$ 和 $p_1$ 间走 Wasserstein 最短路径。

对高斯先验，OT 路径恰好是**线性插值**（在某些情况下）—— 这是 rectified flow 选 linear 的理论支持。

---

### 7.3 其他选择

| 路径 | 公式 | 用在 |
|------|------|------|
| Linear (Rectified) | $x_t = (1-t)x_0 + t x_1$ | SD 3, Pi-0 |
| VP-like | $x_t = \alpha_t x_1 + \sigma_t \epsilon$ | DDPM 等价 |
| OT (Brenier map) | 最优 | 计算上昂贵 |

实践中 linear 用得最多。

---

## §8 SD 3 用的 sigmoid schedule

SD 3 (Esser et al., 2024) 对训练时的 $t$ 采样用了非均匀分布：

```python
def sd3_t_sampler(batch_size):
    # logit-normal sampling
    u = torch.randn(batch_size) * 1.0  # std=1
    t = torch.sigmoid(u)  # in (0, 1)
    return t
```

**动机**：FM 的中等 $t$（$t \approx 0.5$）训练最有挑战，logit-normal 给中间 $t$ 更高采样权重。

类比 DDPM Importance Sampling，但在 FM 框架里实现更简单。

---

## §9 与 VLA / Embodied 的连接（Pi-0 关键）

### 9.1 Pi-0 的 action diffusion

Pi-0 (Physical Intelligence, 2024) 用 Flow Matching **生成 robot actions**：

```
Vision-Language Model (PaliGemma) → backbone features
                ↓
Action Flow Matching Head → trajectory of robot actions
```

为什么 FM 适合机器人？
1. **少步采样**：FM 5-10 步出结果，DDPM 50 步不可接受（机器人要 20Hz 控制）
2. **简洁实现**：直接 MSE，不需要复杂 schedule
3. **连续动作空间天然适配**：action 不像图像有 discretization 问题

---

### 9.2 Action chunking

Pi-0 把"未来 50 步 action"作为一个 trajectory $\mathbf{a}_{1:50} \in \mathbb{R}^{50 \times 7}$，用 FM 一次性生成。

```python
loss = F.mse_loss(v_pred, action_chunk - noise)
```

**关键创新**：action 不是单帧预测，而是一段轨迹的"图像"——FM 像生图一样生 trajectory。

---

## §10 工程实战

### 10.1 训练代码（最简）

```python
def train_step(model, x_1, optimizer):
    """x_1 是真实数据（image latent / action chunk / ...）"""
    t = torch.rand(x_1.shape[0], device=x_1.device)
    x_0 = torch.randn_like(x_1)
    x_t = (1 - t.view(-1, 1, 1, 1)) * x_0 + t.view(-1, 1, 1, 1) * x_1
    v_pred = model(x_t, t)
    loss = F.mse_loss(v_pred, x_1 - x_0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
```

---

### 10.2 推理代码

```python
@torch.no_grad()
def sample(model, n_steps=10):
    x = torch.randn(...)
    timesteps = torch.linspace(0, 1, n_steps + 1)
    for i in range(n_steps):
        t = timesteps[i]
        dt = timesteps[i+1] - t
        v = model(x, t.expand(x.shape[0]))
        x = x + v * dt
    return x
```

可以加 Heun 二阶：
```python
v1 = model(x, t)
v2 = model(x + v1 * dt, t + dt)
x = x + (v1 + v2) / 2 * dt  # 2nd order
```

---

### 10.3 CFG with FM

CFG 在 FM 下：
```python
v_uncond = model(x, t, null_cond)
v_cond = model(x, t, cond)
v = v_uncond + s * (v_cond - v_uncond)
```

—— 完全相同形式，因为 CFG 是 score 空间的线性组合，velocity 也是 score 的线性变换。

---

## §11 课后任务

### 必做
1. 跑通 nb09（2D Flow Matching 玩具实验）
2. 用同一数据集训 DDPM 和 Rectified Flow，比较 NFE-FID Pareto
3. 推导 Linear path 的 $u_t$ = $x_1 - x_0$（不要看讲义）

### 进阶
4. 实现 Heun's method for FM，与 Euler 比较
5. 读 Pi-0 论文，理解 action FM 的输入输出格式
6. 自己设计一个 "L 形路径"（先沿 x 走再沿 y 走），观察 ODE 积分质量

---

## §12 FAQ

**Q1：Flow Matching 比 DDPM 一定好吗？**

A：**不绝对**。
- SD 3 用 FM 击败 SD XL，但 SD XL 也是从 DDPM 路线训出来的
- 小数据（<100K）下 DDPM 仍可能更稳健
- FM 在工程上的优势（简洁、少步采样）有时比纯 FID 数字更重要

---

**Q2：为什么不直接用 OT 路径而要 linear？**

A：OT 路径计算昂贵（需要解 Monge-Kantorovich 问题）。Linear 是"OT 在某些假设下"的精确解，工程上"足够好"。

近期 Rectified Flow 论文 (Liu 2022) 提出"reflow"——把训好的 FM 用作初始化，重新训以"拉直"路径，进一步逼近 OT。

---

**Q3：Flow Matching 能做 inversion（DDIM Inversion 类比）吗？**

A：可以。因为是 ODE，**完全可逆**：
```python
def invert(x_1, n_steps=10):
    x = x_1.clone()
    timesteps = torch.linspace(1, 0, n_steps + 1)  # 反向
    for i in range(n_steps):
        t = timesteps[i]
        dt = timesteps[i+1] - t  # 负的
        v = model(x, t.expand(x.shape[0]))
        x = x + v * dt
    return x  # 这是 x_0
```

且数值上比 DDIM Inversion 更准（无累积近似误差）。

---

## §13 参考资源

- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023（**FM 原论文**）
- Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, ICLR 2023
- Albergo & Vanden-Eijnden, *Building Normalizing Flows with Stochastic Interpolants*, 2023
- Esser et al., *Scaling Rectified Flow Transformers* (SD 3), 2024
- Black et al., *Pi-0: A Vision-Language-Action Flow Model*, 2024

---

> 下一讲（L13）我们再压缩一次：把 ODE 采样压到**单步**——Consistency Models。
