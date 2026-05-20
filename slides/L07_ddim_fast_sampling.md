# Lecture 07｜DDIM 与快速采样：从 1000 步到 50 步

> **本讲目标**
> - 理解 DDIM 的核心想法：非马尔可夫前向过程 + 确定性反向过程
> - 完整推导 DDIM 的更新公式
> - 理解 DDIM 与 probability flow ODE 的关系
> - 入门 DPM-Solver 等高阶 ODE 求解器
> - 掌握采样器评估方法

> **对应论文**：
> - Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021
> - Lu et al., *DPM-Solver: A Fast ODE Solver*, NeurIPS 2022

---

## §1 DDPM 采样为什么慢？

回顾 L04 的采样算法：
```
for t = T, T-1, ..., 1:
    x_{t-1} = ε_θ-based update (x_t, t)
```

每一步**必须**做一次完整的网络前向（U-Net 调用一次）。

**问题**：
- $T = 1000$ 意味着每张图生成需要 1000 次前向
- 在 RTX 3090 上生成单张 512×512 SD 图需要 ~30 秒
- 视频、3D 应用完全无法接受

**核心问题**：能否在质量损失很小的前提下，用 ≤50 步生成？

---

## §2 DDIM 的关键洞察

### 2.1 DDPM 损失函数对前向过程的"无关性"

仔细看 L04 推导的 simplified loss：
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

**只依赖于 $q(x_t | x_0)$ 这一个边缘分布**，不依赖于完整马尔可夫链 $q(x_{1:T} | x_0)$！

> **🔑 DDIM 的关键洞察**：训练好的 DDPM 网络其实是在拟合 score（不依赖采样路径）。我们可以**改变采样路径**，只要保证每个时间点的边缘分布不变。

---

### 2.2 设计非马尔可夫前向过程

DDPM 假设：$q(x_{1:T}|x_0) = \prod_t q(x_t | x_{t-1})$（马尔可夫）

DDIM 改写：$q_\sigma(x_{1:T}|x_0) = q_\sigma(x_T|x_0) \prod_{t=2}^T q_\sigma(x_{t-1} | x_t, x_0)$（非马尔可夫，显式条件于 $x_0$）

要求：
- 每个 $q_\sigma(x_t | x_0)$ 与 DDPM 相同（保证训练目标不变）
- $q_\sigma(x_{t-1} | x_t, x_0)$ 是高斯，但方差 $\sigma_t^2$ 可调

通过繁琐但机械的推导（参考 derive_05），可以得到：

$$q_\sigma(x_{t-1} | x_t, x_0) = \mathcal{N}\left( \sqrt{\bar\alpha_{t-1}} x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\bar\alpha_t} x_0}{\sqrt{1 - \bar\alpha_t}}, \sigma_t^2 I \right)$$

---

### 2.3 DDIM 的两个极端

**当 $\sigma_t^2 = \tilde\beta_t$（DDPM posterior 方差）**：
- 退化为 DDPM 标准采样
- 引入随机性，多样性高

**当 $\sigma_t^2 = 0$（DDIM 默认）**：
- 反向过程**完全确定**（无随机项）
- 给定 $x_T$，输出 $x_0$ 唯一
- 这就是 DDIM

---

## §3 DDIM 更新公式（核心）

把 $\sigma_t^2 = 0$ 代入上式，并用 $x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t} \epsilon_\theta}{\sqrt{\bar\alpha_t}}$（从噪声预测反推 $x_0$）：

$$\boxed{x_{t-1} = \sqrt{\bar\alpha_{t-1}} \cdot \hat x_0(x_t) + \sqrt{1 - \bar\alpha_{t-1}} \cdot \epsilon_\theta(x_t, t)}$$

其中：
$$\hat x_0(x_t) = \frac{x_t - \sqrt{1-\bar\alpha_t} \cdot \epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

**直观解释**：
1. 用网络预测当前 $\epsilon$
2. 反推估计的 $\hat x_0$
3. 用 $\hat x_0$ 沿着 forward 公式"前进"到 $x_{t-1}$

---

### 3.1 跳步采样

DDIM 的**真正威力**：可以选择采样子序列 $\tau_1 < \tau_2 < \dots < \tau_S$（$S \ll T$）。

每步从 $x_{\tau_i}$ 直接跳到 $x_{\tau_{i-1}}$：

$$x_{\tau_{i-1}} = \sqrt{\bar\alpha_{\tau_{i-1}}} \cdot \hat x_0(x_{\tau_i}) + \sqrt{1 - \bar\alpha_{\tau_{i-1}}} \cdot \epsilon_\theta(x_{\tau_i}, \tau_i)$$

**关键**：每次都用同一个训练好的 $\epsilon_\theta$（不用重训）。

> **🔑 这意味着**：训一个 DDPM，可以用任意步数采样！这是真正的 "decouple training from inference"。

---

### 3.2 步数选择

常用 $S$ 值：
- $S = 1000$：与 DDPM 等价（FID 最低）
- $S = 100$：质量基本不变（FID 微涨 0.1-0.3）
- $S = 50$：FID 涨 1-2（视觉差异肉眼不可见）
- $S = 20$：FID 明显变差，但仍可用
- $S = 10$：可见劣化

**SD 的默认配置**：50 步 DDIM 是工业标准。

---

## §4 DDIM 与 Probability Flow ODE 的关系

### 4.1 DDIM 是 ODE 的一阶离散化

回顾 L06 中的 probability flow ODE（VP-SDE 形式）：
$$\frac{dx}{dt} = -\frac{1}{2} \beta(t) x - \frac{1}{2} \beta(t) \cdot \nabla_x \log p_t(x)$$

用 noise 与 score 等价关系 $\nabla_x \log p_t = -\epsilon_\theta / \sqrt{1-\bar\alpha_t}$，并做"参数化变量替换"（详见 derive_05 附录），可以证明：

**DDIM 更新公式精确等价于该 ODE 的一阶 Euler 离散化**（在适当参数化下）。

---

### 4.2 启示

既然 DDIM 是 ODE 求解器，自然可以用**高阶**求解器加速：
- DDIM = 1 阶 Euler
- Heun's method = 2 阶
- DPM-Solver = 多阶 + 解析积分

这就是 DPM-Solver 的出发点。

---

## §5 DPM-Solver（高阶 ODE 求解器）

### 5.1 关键思想：半解析积分

普通 Euler 法：
$$x_{t+\Delta t} \approx x_t + f(x_t, t) \cdot \Delta t$$

这里 $f$ 含 $\epsilon_\theta(x_t, t)$（昂贵），$\Delta t$ 大时误差累积大。

**DPM-Solver 的洞察**：把 ODE 分解为
$$\frac{dx}{dt} = f_{\text{lin}}(t) \cdot x + f_{\text{nonlin}}(x, t)$$

其中线性部分 $f_{\text{lin}}$ **解析可积**。然后只需要数值积分非线性部分。

效果：在 $\Delta t$ 大时仍精确——**10 步 DPM-Solver 可媲美 50 步 DDIM**。

---

### 5.2 DPM-Solver 算法（一阶版本）

```python
# DPM-Solver-1 (相当于 DDIM 但参数化不同)
def dpm_solver_1_step(x, t, t_prev, model):
    lambda_t = lambda_func(t)        # log(α/σ) 的某种参数化
    lambda_prev = lambda_func(t_prev)
    h = lambda_prev - lambda_t

    eps = model(x, t)
    x_prev = (alpha_prev / alpha_t) * x - sigma_prev * (math.exp(h) - 1) * eps
    return x_prev
```

二阶/三阶版本添加额外的中间评估，类似 Runge-Kutta。

**实际表现**（CIFAR-10 FID）：

| Sampler | 10 NFE | 20 NFE | 50 NFE |
|---------|--------|--------|--------|
| DDPM | - | 32.6 | 7.5 |
| DDIM | 13.4 | 6.8 | 4.7 |
| DPM-Solver-2 | 4.7 | 2.8 | 2.7 |
| DPM-Solver-3 | 3.6 | 2.7 | 2.6 |

NFE = Number of Function Evaluations（神经网络前向次数）

---

## §6 其他 Sampler 简介

### 6.1 Euler / Euler Ancestral

最朴素的 Euler 离散化。`Euler Ancestral` 加入采样噪声（介于 DDIM 和 DDPM 之间）。

### 6.2 Heun's Method

2 阶 ODE 求解器（Karras 2022 EDM 默认）：
```
预测一步 → 用平均梯度修正
```

### 6.3 UniPC

Lu 2023 提出的统一 predictor-corrector，进一步压缩到 5-8 步。

### 6.4 总结：选择哪个？

| 场景 | 推荐 |
|------|------|
| 学习/复现论文 | DDIM（最简单，最经典） |
| SD 推理 | DPM-Solver++ 2M（diffusers 默认） |
| 极少步推理（<10步） | UniPC, DPM-Solver++ |
| 需要确定性 | 任何 ODE-based（不要 ancestral） |
| 需要多样性 | DDPM, Euler Ancestral |

---

## §7 评估采样器

### 7.1 评估指标

衡量采样器好坏的三个维度：

1. **质量**：FID, IS（在固定步数下）
2. **速度**：NFE（神经网络前向次数）
3. **多样性**：generated diversity, recall

### 7.2 标准 protocol

固定**训练好的模型**和**随机种子**，扫描不同步数：
```
for sampler in [ddim, dpm_solver, ...]:
    for nfe in [10, 20, 50]:
        FID = evaluate(sampler, nfe, model, seed=42)
        record(sampler, nfe, FID)
```

绘制 Pareto 曲线（FID vs NFE）：
```
FID
 │
 │ ┃ DDPM
 │ ┃
 │ ╲ DDIM
 │  ╲╲
 │   ╲╲ DPM-Solver
 │    ──────────
 └───────────────── NFE
```

---

## §8 工程实战要点

### 8.1 数值稳定性

DDIM 的 $\hat x_0$ 计算涉及除以 $\sqrt{\bar\alpha_t}$，当 $t$ 接近 $T$ 时 $\bar\alpha_t \to 0$，**除零风险**。

实践：
- 把 $\hat x_0$ clip 到合理范围（如 $[-3, 3]$ for normalized images）
- 不直接计算 $\hat x_0$ 中间量，用代数化简后的形式

### 8.2 时间步选择

跳步采样的时间序列选择有讲究：
- **Linear**：均匀间隔 → 简单，但前期粗后期精
- **Quadratic**：$\tau_i \propto i^2$ → 前期密集，后期稀疏
- **Karras**：基于 SNR 的非线性间隔 → SOTA

Karras 时间步：
```python
sigma_max, sigma_min = 80.0, 0.002
rho = 7.0
sigmas = (sigma_max**(1/rho) + ramp * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
```

### 8.3 Inference 的 batch 设计

DDIM 推理时，每步同时处理 batch。建议：
- batch_size 取决于显存而非速度
- 不同 sample 的 $t$ 序列**相同**（与训练时不同）
- 可以预先计算所有时间步的系数

---

## §9 DDIM Inversion（前向技巧）

### 9.1 反向问题

给定一张真实图像 $x_0$，能否找到一个 $x_T \sim \mathcal{N}(0, I)$ 使得 DDIM 采样能精确还原 $x_0$？

**答案**：对 DDIM（确定性）可以！这叫 **DDIM Inversion**。

### 9.2 算法

把 DDIM 的反向公式**反着用**：
$$x_t = \sqrt{\bar\alpha_t} \cdot \hat x_0(x_{t-1}) + \sqrt{1-\bar\alpha_t} \cdot \epsilon_\theta(x_{t-1}, t-1)$$

注意是用 $\epsilon_\theta(x_{t-1}, t-1)$ 而不是 $\epsilon_\theta(x_t, t)$——这是 **一阶近似**，有累积误差。

### 9.3 应用

- **图像编辑**：先 invert，修改条件，再 sample
- **风格迁移**：保留 $x_T$，换 prompt
- **Null-text inversion**（Mokady 2022）：更精确的 inversion 技术

> **🔑 SD 中的 img2img 功能**就基于 DDIM inversion（局部）。

---

## §10 本讲核心要点

1. **DDIM 把训练与采样解耦**：训一次，任意步数推理
2. **DDIM 更新公式**：$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \hat x_0 + \sqrt{1-\bar\alpha_{t-1}} \epsilon_\theta$（必背）
3. **DDIM = 一阶 ODE 离散化**，对应 probability flow ODE
4. **DPM-Solver** 利用半解析积分，10 步可达 DDIM 50 步效果
5. **采样器选择**取决于场景：质量优先用高阶，简单优先用 DDIM
6. **DDIM Inversion** 让确定性采样可逆，是图像编辑的基础

---

## §11 课后任务（Project 2 启动）

### 必做（Project 2 基础档）

1. **代码实现**：用 Project 1 训练好的 DDPM 模型，实现 DDIM sampler
   - 不需要重新训练
   - 实现跳步采样（支持任意步数 S）
   - 对比 DDIM 50 步 vs DDPM 1000 步的生成质量
   - 提交 `ddim_sampler.py` + 对比报告

2. **思考题**：
   - DDIM 为什么"确定性"？给定相同 $x_T$，输出是否一定相同？
   - 如果用 DDIM 100 步推理一个**用 cosine schedule** 训练的模型，时间步如何选？
   - 为什么 $\sigma_t^2$ 在 DDIM 和 DDPM 之间"插值"会得到不同质量？

### 进阶档

3. **DPM-Solver-2 实现**：阅读 Lu 2022 论文 §3，实现 2 阶版本
   - 在 CIFAR-10 上扫描 NFE = {10, 20, 50}
   - 画 Pareto 曲线（DDIM vs DPM-Solver-2）

### 挑战档

4. **DDIM Inversion**：实现 inversion 算法，做一个简单的图像编辑 demo

---

## §12 推荐进一步阅读

| 资源 | 重点 |
|------|------|
| DDIM 论文 | §3 必读 |
| DPM-Solver 论文 | §3 算法部分 |
| Karras 2022 EDM | 工程化采样器最佳实践 |
| diffusers 库的 schedulers | 实战代码参考 |
| 推导手稿 `derive_05_ddim.pdf` | 完整非马尔可夫推导 |

---

## §13 常见问题

**Q: DDIM 是 SDE 还是 ODE？**

A: 当 $\sigma_t^2 = 0$ 时是 ODE（确定性）；当 $\sigma_t^2 \neq 0$ 时是某种 SDE。论文里的 "DDIM" 默认指前者。

**Q: 为什么 DDIM 在 SD 里只用 50 步而不是 20 步？**

A: 工程权衡。20 步 FID 不算太差，但生成图细节会损失（手指、文字这类高频内容）。50 步是个 sweet spot。

**Q: 我能不能直接训 DDIM 而不训 DDPM？**

A: 不可以。DDIM 没有自己的训练目标，它只是 DDPM 的另一种采样方式。所谓"DDIM model"其实就是 DDPM model。

**Q: 用 DDIM Inversion 后我能精确恢复原图吗？**

A: 不能 100% 精确。一阶近似有累积误差。在 50 步下，重构误差约 1-3% PSNR loss。如需高精度 inversion，用 null-text inversion 或 prompt-tuning。

**Q: DPM-Solver 比 DDIM 好这么多，为什么不所有人都用？**

A: 三个原因：
1. DDIM 实现简单，DPM-Solver 算法和数值细节复杂
2. DPM-Solver 在某些 SD 微调场景下不如 DDIM 稳定
3. 历史原因：早期教程都用 DDIM，惯性大

但产业实践中 DPM-Solver / DPM-Solver++ 已经是 SD 默认（diffusers, AUTOMATIC1111）。

---

> **下一讲预告**：L08 进入条件生成的世界——**Classifier Guidance** 与 **Classifier-Free Guidance (CFG)**。CFG 是 Stable Diffusion、Imagen 等所有现代 text-to-image 模型的基础工具。
