# Score SDE 统一图景：如何读懂 §8 那张图

这份补充阅读对应 `slides/L06_score_sde.md` 的 §8「一个统一图景」，逐层拆解那张图，并给出一个通用的读图方法：**以后看到任何新的 diffusion sampler 论文，都可以先问"它是在图里的哪条边上做文章"。**

先回顾原图：

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

## 1. 顶层：Score SDE 框架

所有扩散模型（无论叫 DDPM 还是 NCSN）本质上都在做同一件事：给数据加噪声，让它变成一条随机过程 $x_t$，然后学习每个时刻的 score

$$
\nabla_x \log p_t(x)
$$

再用 Anderson 公式（见 `slides/L06_score_sde.md` §4，反向 SDE：

$$
dx = \left[ f(x, t) - g(t)^2 \nabla_x \log p_t(x) \right] dt + g(t)\, d\bar W
$$

）把噪声反着"score-guide"回数据。这是整张图里最抽象、最统一的一层，DDPM 和 NCSN 都只是它的特例。

## 2. 第二层：两条分支 = 两种"加噪方式"的选择

Score SDE 只是一个通用模板

$$
dx = f(x,t)\,dt + g(t)\,dW
$$

具体怎么加噪，取决于怎么选 $f, g$（对照 `slides/L06_score_sde.md` §3.3 的表格）：

| 性质 | VP-SDE（DDPM） | VE-SDE（NCSN） |
|------|----------------|----------------|
| Drift $f$ | $-\frac12\beta(t)x$（向原点收缩） | $0$ |
| Diffusion $g$ | $\sqrt{\beta(t)}$ | $\sqrt{d\sigma^2/dt}$ |
| $\text{Var}(x_T)$ | 有界（→ 1） | 无界（→ ∞） |
| 终态分布 | $\mathcal{N}(0, I)$ | $\mathcal{N}(0, \sigma_T^2 I)$，$\sigma_T$ 极大 |

- **VP-SDE**（Variance Preserving）：$dx = -\frac12\beta(t)x\,dt + \sqrt{\beta(t)}\,dW$。均值被 drift 项持续拉向 $0$，同时注入噪声，方差始终有界。这是 DDPM 单步公式 $x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon$ 取连续极限（$T \to \infty$，$\beta_t = \beta(t)\Delta t$）得到的（§3.1）。
- **VE-SDE**（Variance Exploding）：$dx = \sqrt{d\sigma^2(t)/dt}\,dW$。没有 drift，均值不动，只是不断叠加越来越大的噪声，方差随时间无界增长。这是 NCSN 多噪声尺度训练方式（$\sigma_1 > \sigma_2 > \dots$）连续化后的极限（§3.2）。

**这一步的含义**：DDPM 和 NCSN 表面上像是两套不相关的方法——一个预测噪声 $\epsilon$，一个估计 score——但其实只是同一个 SDE 框架里选了不同的 $(f, g)$。

## 3. 第三层：把连续 SDE 离散化 = 具体算法

确定了 SDE 之后，要在计算机上采样就得把连续时间离散成有限步：

- VP-SDE 离散化 → **DDPM**，用的是 **ancestral sampling**（从 $x_T$ 一步步采样回 $x_0$，对应课程 W2-W4 的算法）。
- VE-SDE 离散化 → **NCSN**，用的是 **annealed Langevin dynamics**（在每个噪声尺度上跑若干步 Langevin MCMC，噪声尺度逐渐"退火"变小）。

换句话说，DDPM 和 NCSN 这两个历史上看起来完全独立发展出来的方法，其实是同一个反向 SDE 用两种不同离散化方式得到的特例（对应 `slides/L06_score_sde.md` §9 要点 2）。

## 4. 第四层：两条分支重新汇合到 Probability Flow ODE

不管从 VP-SDE 还是 VE-SDE 出发，都可以用 Anderson 公式的"去随机性版本"（§5.1，详见 [Probability Flow ODE 补充阅读](probability-flow-ode-marginals.md)）把 SDE 改写成一个确定性 ODE：

$$
\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)
$$

它在每个时间 $t$ 的边缘分布 $p_t(x)$ 与原 SDE 完全相同，但轨迹不再含随机噪声项。这个 probability flow ODE 才是后续所有"快速采样器"的共同祖先：

- 一阶离散化 → **DDIM**
- 更高阶、更聪明的数值方法 → **DPM-Solver**、**EDM**

## 5. 怎么用这张图读论文

每条箭头都代表一次 reduction 或近似：

| 箭头 | 对应的化简/近似 |
|---|---|
| 离散 DDPM ↔ 连续 VP-SDE | 连续极限（$T\to\infty$，$\beta_t=\beta(t)\Delta t$） |
| Score SDE 框架 → VP/VE-SDE | 选择加噪方式的 $(f, g)$ |
| VP/VE-SDE → DDPM/NCSN | 离散化方式（ancestral sampling / annealed Langevin） |
| SDE → Probability Flow ODE | 去随机性（Anderson 公式的确定性版本） |

以后看到一篇新论文说"我们提出了一种新的扩散采样器"，可以直接对照这张图问：

1. 它是不是换了一种新的加噪方式？——即在 VP/VE 之外提出了新的 $(f, g)$（比如 EDM 的参数化）。
2. 它是不是换了离散化/数值求解器？——比如用更高阶的 ODE solver（DPM-Solver、Heun）替代一阶 Euler。
3. 它是不是换了 predictor-corrector 的组合方式？——比如加了额外的 Langevin 校正步（`slides/L06_score_sde.md` §6）。

这三个问题基本能把绝大多数"新 sampler"论文迅速定位到图上的某一条边，而不必把它当成一个孤立的新发明。
