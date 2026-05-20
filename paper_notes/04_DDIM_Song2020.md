# 论文导读 04: DDIM (Song et al., ICLR 2021)

**标题**：Denoising Diffusion Implicit Models
**作者**：Jiaming Song, Chenlin Meng, Stefano Ermon (Stanford)
**核心地位**：⭐⭐⭐⭐⭐ **工业级扩散模型基础**。把采样从 1000 步降到 50 步

---

## 一、为什么必读

如果说 DDPM 是研究界的扩散基础，DDIM 是**工业界的扩散基础**。
- 你用过 Stable Diffusion → 用过 DDIM
- "训练一次 inference 多种方式" 的灵感来源
- 后续 DPM-Solver、Consistency Model 都站在 DDIM 肩上

---

## 二、阅读路线图

| 章节 | 优先级 | 备注 |
|------|--------|------|
| §1 Intro | ✅ | 动机：DDPM 慢 |
| §2 Background | 跳过 | 已学 DDPM |
| **§3 Variational inference for non-Markovian forward processes** | 🔴 核心 | 整篇精华 |
| §4.1 Sampling | 🔴 必读 | DDIM update 公式 |
| §4.2 Accelerated generation | 🔴 必读 | 跳步采样 |
| §4.3 Relevance to neural ODEs | ✅ | 与 ODE 的连接 |
| §5 Experiments | ✅ | 看 NFE-FID 表格 |

---

## 三、核心公式

### Non-Markovian Forward
$$q_\sigma(x_{t-1} | x_t, x_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}} x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2} \cdot \epsilon, \sigma_t^2 I)$$

其中 $\epsilon$ 由 $x_t, x_0$ 决定。

### DDIM Update（$\sigma=0$）
$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \hat x_0(x_t) + \sqrt{1-\bar\alpha_{t-1}} \epsilon_\theta(x_t, t)$$

其中 $\hat x_0(x_t) = (x_t - \sqrt{1-\bar\alpha_t} \epsilon_\theta) / \sqrt{\bar\alpha_t}$。

---

## 四、关键洞察

1. **训练目标只依赖 $q(x_t|x_0)$**：所以可改 forward 链而不影响训练（**核心 trick**）
2. **$\sigma_t = 0$ 得到确定性 DDIM**：可逆 ODE
3. **跳步采样自由**：选任意时间步子序列即可，无需重训

---

## 五、易混淆点

### 1. "DDIM 是模型还是 sampler？"

**是 sampler**。"DDIM 模型"其实就是 DDPM 模型——同一个 $\epsilon_\theta$ 网络。

---

### 2. "DDIM 严格快于 DDPM 吗？"

**FID 上不一定**。在同样的 NFE 下：
- $N \geq 100$：DDIM ≈ DDPM
- $N \leq 50$：DDIM 显著好（因为可跳步）
- $N = 1000$：DDPM 更好（DDIM 失去随机性优势）

DDIM 的优势在**少步数**，不是绝对优。

---

### 3. "DDIM Inversion 能 100% 还原图像？"

**不能**。Inversion 用了**一阶近似**（$\epsilon_\theta(x_{t-1}, t-1)$ 替代 $\epsilon_\theta(x_t, t)$），有累积误差。50 步典型重构误差 1-3% PSNR。

---

### 4. "$\sigma_t$ 在 DDPM 和 DDIM 之间任意插值？"

**理论上**：是的（$\sigma_t^2 = \eta \tilde\beta_t$，$\eta \in [0, 1]$）。
**实践上**：要么 $\eta=0$（DDIM），要么 $\eta=1$（DDPM）。中间值用得不多。

---

## 六、与其他论文的关系

- **DDPM (Ho 2020)**：直接基于其训练 → 推理时改 sampler
- **Score SDE (Song 2021)**：同一第一作者的并行工作。DDIM 是 probability flow ODE 的一阶离散化
- **DPM-Solver (Lu 2022)**：DDIM 的高阶推广
- **EDM (Karras 2022)**：把 DDIM 在自己的统一框架中重新写

---

## 七、必看实验表

| Sampler | NFE | CIFAR-10 FID |
|---------|-----|--------------|
| DDPM | 1000 | 4.04 |
| DDIM | 1000 | 4.16 |
| DDIM | 100 | 4.20 |
| DDIM | 50 | 4.67 |
| DDIM | 20 | 6.84 |
| DDIM | 10 | 13.36 |

观察：**100 步几乎无损，50 步可用，20 步显著降质**。SD 默认 50 步是这表的延续。

---

## 八、思考题

1. DDIM 公式有个对应的"确定性 ODE"：$\frac{dy}{d\tau} = \epsilon_\theta$（参数化变量替换后）。这个 ODE 对应 probability flow ODE 在什么变量替换下的形式？
2. 为什么 DDIM 步数从 50 降到 20，质量损失就突然变大？画 SNR 曲线分析
3. DDIM 的 $\sigma_t = 0$ 与 $\sigma_t = \tilde\beta_t$ 之间，理论上是否存在"最佳"$\sigma$？

---

## 九、引用

```
@inproceedings{song2021denoising,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  booktitle={ICLR},
  year={2021}
}
```
