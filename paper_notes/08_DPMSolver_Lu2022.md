# 论文导读 08: DPM-Solver (Lu et al., NeurIPS 2022)

**标题**：DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps
**作者**：Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu (THU)
**核心地位**：⭐⭐⭐⭐ **工业级 sampler 之王**。10 步采样达到 DDIM 50 步质量

---

## 一、为什么必读

- diffusers 库的默认 sampler（DPMSolverMultistepScheduler）
- 推理速度提升 5x，质量几乎无损
- 把 ODE 数值积分的 90 年代理论应用到深度生成
- 后续 UniPC、DPM-Solver++ 都是衍生

---

## 二、阅读路线图

| 章节 | 优先级 |
|------|--------|
| §1 Intro | ✅ |
| §2 Background | 跳过 |
| **§3 DPM-Solver** | 🔴 核心 |
| §3.1 ODE reformulation | 🔴 核心 |
| §3.2 Exact solution | 🔴 核心 |
| §3.3 Approximation (DPM-Solver-k) | 🔴 核心 |
| §4 Experiments | ✅ |

---

## 三、核心思想

### 关键洞察

DDIM 是一阶 Euler 离散化的 probability flow ODE。如果用**高阶 ODE 求解器**会更精确。

但直接用 RK45 等通用求解器**不够好**——因为 diffusion ODE 有特殊结构（线性 drift + 非线性 score）。

### 半解析积分

把 ODE 写成：
$$\frac{dx}{dt} = f(t) \cdot x + g(t) \cdot \epsilon_\theta(x, t)$$

**线性部分** $f(t) \cdot x$ **解析可积**。只对非线性部分做数值积分。

最终得到（在参数变换 $\lambda = \log(\alpha/\sigma)$ 下）：

$$x_{\lambda_{i+1}} = \frac{\alpha_{\lambda_{i+1}}}{\alpha_{\lambda_i}} x_{\lambda_i} - \alpha_{\lambda_{i+1}} \int_{\lambda_i}^{\lambda_{i+1}} e^{-\lambda} \epsilon_\theta(\hat x_\lambda, t_\lambda) \, d\lambda$$

只需要数值积分这个右侧积分。

---

## 四、Solver 阶数

### DPM-Solver-1（≡ DDIM）
一阶近似积分项。每步 NFE=1。

### DPM-Solver-2（midpoint method）
在中点 $\lambda_{\text{mid}} = (\lambda_i + \lambda_{i+1})/2$ 处再评估一次。每步 NFE=2。

### DPM-Solver-3
更高阶，每步 NFE=3。

**Trade-off**：更高阶 → 更精确 → 但每步更贵。最优阶数取决于总 NFE budget。

---

## 五、实验关键表

| NFE | DDIM FID | DPM-Solver-2 | DPM-Solver-3 |
|-----|----------|--------------|--------------|
| 10 | 13.36 | 4.65 | 3.55 |
| 20 | 6.84 | 2.87 | 2.65 |
| 50 | 4.67 | 2.69 | - |

**惊人发现**：DPM-Solver-3 在 NFE=10 时 FID=3.55，**优于** DDIM 在 NFE=50 时的 4.67。

---

## 六、易混淆点

### 1. "DPM-Solver 适用于所有扩散模型？"

**适用于任何用 noise prediction 的扩散模型**。所以：
- DDPM、SD、SDXL：全部适用
- Flow Matching、Consistency Models：不适用（参数化不同）

---

### 2. "DPM-Solver 与 DDIM 哪个更稳定？"

**两难权衡**：
- DDIM 一阶，数值稳定，几乎不会失败
- DPM-Solver 高阶，在大 NFE 下精确，但小 NFE 下偶尔有数值问题

SD 实践中：DPM-Solver++ 2M (multistep variant) 是默认。

---

### 3. "为什么 multistep 比 singlestep 好？"

**Singlestep** (论文原版)：每步独立用前一步信息。
**Multistep** (DPM-Solver++): 用最近几步的 history（类似 BDF formula）。

Multistep 在每步 NFE=1 的情况下也能达到高阶精度。

---

## 七、与其他论文的关系

- **DDIM (Song 2021)**：一阶 DPM-Solver
- **Score SDE (Song 2021)**：DPM-Solver 是其 ODE 视角的工程化
- **DPM-Solver++ (Lu 2022)**：本文的改进版，multistep + guided
- **UniPC (Zhao 2023)**：统一 predictor-corrector，进一步压缩到 5-8 步
- **EDM-Heun (Karras 2022)**：另一种二阶 sampler

---

## 八、工程实战

### 在 diffusers 中
```python
from diffusers import DPMSolverMultistepScheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# 推荐配置:
# solver_order=2, prediction_type='epsilon', algorithm_type='dpmsolver++'
```

### 步数推荐
- 5-10 步：DPM-Solver-2/3 multistep
- 15-25 步：DPM-Solver-2 单步
- 30+：DDIM 也够好

---

## 九、思考题

1. 为什么参数化 $\lambda = \log(\alpha/\sigma)$ 而非 $t$ 直接？这种变换的几何含义？
2. DPM-Solver-2 与 RK2 的区别是什么？为什么不直接用 RK2？
3. CFG 与 DPM-Solver 结合时，每步需要 NFE=2×k（k 是 solver 阶数）。这是否抵消了 DPM-Solver 的速度优势？

---

## 十、引用

```
@inproceedings{lu2022dpm,
  title={DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps},
  author={Lu, Cheng and Zhou, Yuhao and Bao, Fan and Chen, Jianfei and Li, Chongxuan and Zhu, Jun},
  booktitle={NeurIPS},
  year={2022}
}
```
