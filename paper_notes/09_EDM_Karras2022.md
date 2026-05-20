# 论文导读 09: EDM (Karras et al., NeurIPS 2022)

**标题**：Elucidating the Design Space of Diffusion-Based Generative Models
**作者**：Tero Karras, Miika Aittala, Timo Aila, Samuli Laine (NVIDIA)
**核心地位**：⭐⭐⭐⭐⭐ **工程师的扩散圣经**。系统性梳理扩散模型设计空间

---

## 一、为什么必读

如果你要**训练自己的扩散模型**，这是除了 DDPM 之外必读的第二篇。
- 系统 ablation 所有设计选择（schedule、loss、preconditioning、sampler）
- 提出**统一框架**，把 DDPM、Score SDE、Improved DDPM 都重新写
- 工程上 SOTA 的 sampler（Heun's method + 自定义 sigma schedule）

> 论文写得像工程手册，每个选择都有 ablation 支撑。

---

## 二、阅读路线图

| 章节 | 优先级 |
|------|--------|
| §1 Intro | ✅ |
| **§2 Improvements to deterministic sampling** | 🔴 核心 |
| §2.1 Reframing | 🔴 核心（统一框架） |
| §2.2 Sigma schedule | 🔴 核心 |
| §2.3 Heun's method | 🔴 核心 |
| §3 Stochastic sampling | ✅ |
| **§4 Score matching reformulation** | 🔴 核心 |
| §4.1 Preconditioning | 🔴 核心 |
| §4.2 Loss weighting | ✅ |
| §5 Experiments | ✅ |

---

## 三、统一框架（§2.1 必看）

EDM 写法：
$$x_t = D_\theta(c_{\text{noise}}(t) \cdot \epsilon \cdot c_{\text{in}}(t) + c_{\text{skip}}(t) \cdot c_{\text{out}}(t) \cdot x_t, t)$$

通过选不同的 $c_{\text{in}}, c_{\text{out}}, c_{\text{skip}}, c_{\text{noise}}$ 函数，可以**重现** DDPM、Score SDE 等所有变体。

**优势**：从一个统一形式出发，每个设计选择都可以系统消融。

---

## 四、Karras Sigma Schedule（§2.2）

$$\sigma_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho})\right)^\rho$$

参数：
- $\sigma_{\min} = 0.002$
- $\sigma_{\max} = 80$
- $\rho = 7$

**对比**：DDPM 的 linear schedule 在低分辨率合理，但 EDM schedule 更精细控制 SNR 分布。

> diffusers 库的 `EulerDiscreteScheduler` 用的就是这个。

---

## 五、Heun's Method（§2.3）

二阶 ODE 求解器：
```python
def heun_step(x, sigma_i, sigma_next, denoise_fn):
    # Step 1: Euler 预测
    d_cur = (x - denoise_fn(x, sigma_i)) / sigma_i
    x_next_euler = x + (sigma_next - sigma_i) * d_cur

    # Step 2: 在新点再评估一次
    d_next = (x_next_euler - denoise_fn(x_next_euler, sigma_next)) / sigma_next

    # Step 3: 修正（平均两个导数）
    x_next = x + (sigma_next - sigma_i) * (d_cur + d_next) / 2
    return x_next
```

每步 NFE=2，比 DDIM 精确。

---

## 六、Preconditioning（§4.1 必看）

让网络预测 $D_\theta$（"denoised image"）而非 $\epsilon$。具体：

$$D_\theta(x, \sigma) = c_{\text{skip}}(\sigma) \cdot x + c_{\text{out}}(\sigma) \cdot F_\theta(c_{\text{in}}(\sigma) \cdot x, c_{\text{noise}}(\sigma))$$

其中：
- $c_{\text{in}} = 1/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$：让输入方差 ≈ 1
- $c_{\text{out}} = \sigma \sigma_{\text{data}} / \sqrt{\sigma^2 + \sigma_{\text{data}}^2}$
- $c_{\text{skip}} = \sigma_{\text{data}}^2 / (\sigma^2 + \sigma_{\text{data}}^2)$
- $c_{\text{noise}} = \frac{1}{4} \log \sigma$

**动机**：让网络无论在大 $\sigma$ 还是小 $\sigma$，输入/输出都接近单位方差。神经网络在这种归一化下学得更好。

---

## 七、关键实验

CIFAR-10 SOTA（NFE=35）：

| 方法 | FID |
|------|-----|
| DDPM | 3.17 |
| Score SDE (PC) | 2.41 |
| **EDM** | **1.79** |

ImageNet 64×64 SOTA（NFE=79）：

| 方法 | FID |
|------|-----|
| ADM-G (CG) | 1.83 |
| **EDM** | **1.36** |

—— EDM 是当时 SOTA。

---

## 八、容易误读

### 1. "EDM 是新模型？"

**不是**。EDM 是一套**训练 + 推理的工程方法**，可以应用于任何 noise prediction 网络。其核心是：
- 用 EDM 的 preconditioning 重新训
- 用 EDM 的 sigma schedule + Heun 推理

---

### 2. "EDM 比 DPM-Solver 好？"

**不直接对比**。EDM 是从头训新模型，DPM-Solver 是给已训好模型加 sampler。
工程上：
- 训新模型 → 直接用 EDM 路线（preconditioning + Heun）
- 已有 SD 模型 → 用 DPM-Solver++ inference

---

### 3. "Preconditioning 一定要 $\sigma_{\text{data}} = 0.5$？"

实验值。$\sigma_{\text{data}}$ 是训练数据的标准差（在 normalize 后接近 0.5）。如果你的数据归一化方式不同，要重新算。

---

## 九、与其他论文的关系

- **Score SDE (Song 2021)**：被 EDM 重新写
- **DDPM/Improved DDPM**：作为 EDM 框架的特例
- **Consistency Models (Song 2023)**：基于 EDM 框架训练
- **SD 3 / Flux**：部分用 EDM 思路

---

## 十、思考题

1. 为什么 $\rho = 7$ 而不是 1, 2, 5？画 sigma schedule 曲线分析
2. Preconditioning 让网络输出 $D_\theta$ 而非 $\epsilon$。两种参数化在数学上等价吗？为什么 EDM 选 $D_\theta$？
3. 如果不用 EDM preconditioning，单换 sigma schedule，能 work 吗？

---

## 十一、推荐使用场景

| 场景 | 用 EDM？ |
|------|---------|
| 从零训扩散模型 | 强烈推荐（preconditioning + schedule） |
| 用预训练 SD 推理加速 | 不直接用（用 DPM-Solver） |
| 复现学术 paper | 看 paper 用什么 |
| 工业部署 | 与 TensorRT/onnx 兼容性需测试 |

---

## 十二、引用

```
@inproceedings{karras2022elucidating,
  title={Elucidating the Design Space of Diffusion-Based Generative Models},
  author={Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  booktitle={NeurIPS},
  year={2022}
}
```
