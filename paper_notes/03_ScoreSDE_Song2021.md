# 论文导读 03: Score SDE (Song et al., ICLR 2021)

**标题**：Score-Based Generative Modeling through Stochastic Differential Equations
**作者**：Yang Song, Jascha Sohl-Dickstein, Diederik Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole
**核心地位**：⭐⭐⭐⭐⭐ **课程数学高峰**。统一 DDPM 与 NCSN 的关键论文

---

## 一、为什么必读

如果说 DDPM 是 W2-W4 的核心论文，Score SDE 就是 W5-W6 的核心论文。
- 提出 **SDE 视角**统一所有扩散模型
- 引入 **probability flow ODE**——DDIM、DPM-Solver 等加速 sampler 的母版
- 提出 **predictor-corrector sampler**，CIFAR-10 上 SOTA

> 学完这篇，你看任何"基于扩散的生成模型"论文都能立刻识别它的归属（VP-SDE、VE-SDE 还是其他变体）。

---

## 二、阅读路线图（关键！）

这是一篇**长论文**（24 页 + 大量附录）。**不要一字一句读完**。建议：

| 优先级 | 章节 | 时间 |
|-------|------|------|
| 🔴 必读 | §1 Intro | 15 min |
| 🔴 必读 | §3 SDE-based generative modeling | 1 hr |
| 🔴 必读 | §3.2 Reverse SDE | 30 min |
| 🔴 必读 | §3.3 Probability flow ODE | 30 min |
| 🔴 必读 | §4 PC sampler (高层) | 30 min |
| ⚠️ 选读 | §5 Architecture & results | 必要时翻 |
| ⚠️ 选读 | Appendix A (proofs) | 推导有兴趣再看 |

**第一次读**：跳过 §2（已学），跳过附录，聚焦 §3 + §3.2 + §3.3。

**第二次读**（深入）：补 §4 + Appendix A 的关键推导。

---

## 三、核心公式

### VP-SDE（对应 DDPM）
$$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dW$$

### VE-SDE（对应 NCSN）
$$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, dW$$

### Reverse SDE
$$dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t) \, d\bar W$$

### Probability Flow ODE
$$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$$

---

## 四、关键洞察（必背）

1. **DDPM ↔ VP-SDE 离散化**：$T \to \infty$ 极限下 DDPM 就是 VP-SDE
2. **NCSN ↔ VE-SDE 离散化**：同上
3. **Score 是反向 SDE 唯一未知量**：所以所有扩散模型本质上都在学 score
4. **Probability flow ODE**：一个**确定性 ODE**给出与 SDE 相同的边缘分布

---

## 五、常被误读

### 1. "SDE 视角是不是只是 DDPM 的数学包装？"

**不是**。SDE 视角实质性扩展了模型能力：
- 允许在连续时间任意 step（不只是 $T$ 个整数 step）
- 允许更高阶的 sampler（DPM-Solver, EDM 等都依赖 ODE 视角）
- 允许严格的 likelihood 计算

---

### 2. "VE-SDE 与 VP-SDE 谁更好？"

**没有统一答案**。
- VP-SDE：方差保持 ≈ 1，工程上数值稳定（SD、Imagen 等都用 VP）
- VE-SDE：理论上简洁（drift = 0），但 $\sigma$ 可以非常大，数值上要小心

EDM (Karras 2022) 论文论证：**两者都不是最优**，应该重新设计 schedule（EDM 公式）。

---

### 3. "Probability flow ODE = SDE 的近似？"

**不是近似，是精确等价**。两者在每个时间 $t$ 的边缘分布完全相同，只是轨迹不同（SDE 随机、ODE 确定）。

---

## 六、与其他论文的关系

- **DDPM, NCSN**：被统一的两个特例
- **DDIM (Song 2021)**：同一作者的并行工作，DDIM = probability flow ODE 的一阶离散化
- **EDM (Karras 2022)**：Score SDE 的工程化重写
- **Flow Matching (Lipman 2023)**：抛弃 SDE 直接学 ODE 向量场——某种意义上是 Score SDE 的极致简化

---

## 七、容易卡壳的数学

### Anderson 公式（§A.1）

第一次读会觉得"凭空冒出来"。其实推导思路是：
1. 写出正向 FPE
2. 假设反向 SDE $dx = \tilde f dt + \tilde g d\bar W$
3. 要求反向 FPE 与正向 FPE 一致
4. 用 $\Delta p = \nabla \cdot (p \nabla \log p)$ 这个**关键恒等式**做整理
5. 得到 $\tilde f = f - g^2 \nabla \log p$

如果卡住，看本课程 derive_04 §3。

---

### Probability flow ODE（§3.3）

类似思路：
- ODE $dx/dt = h(x, t)$ 对应 transport equation
- 要求 transport equation 等于 FPE
- 同样用上面恒等式，得 $h = f - \frac{1}{2} g^2 \nabla \log p$

---

## 八、PC Sampler（§4 速读）

简单理解：
- **Predictor**：用 reverse SDE 走一步
- **Corrector**：用 Langevin MCMC 在当前 $t$ "矫正"
- 一步预测 + K 步矫正

效果：FID 在 CIFAR-10 上 2.41（碾压当时所有方法）。

实践中：
- PC sampler 步数翻倍（昂贵）
- DPM-Solver 等后续方法在更少步数下达到类似 FID
- PC 已较少使用，但概念仍重要

---

## 九、思考题

1. 为什么 $T$（时间步数）在 SDE 视角下"不重要"？这给我们什么启示？
2. 如果用 BDF（backward differentiation formula）等高阶隐式求解器解反向 SDE，会比 PC sampler 更快吗？
3. 论文 Figure 1 给出 SDE-based 模型的整体框架图。请用自己的话复述每个箭头的含义。

---

## 十、引用

```
@inproceedings{song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Song, Yang and Sohl-Dickstein, Jascha and Kingma, Diederik P and Kumar, Abhishek and Ermon, Stefano and Poole, Ben},
  booktitle={ICLR},
  year={2021}
}
```
