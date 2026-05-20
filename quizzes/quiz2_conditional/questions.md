# Quiz 2: 条件生成与 LDM（W5-W9）

> **时长建议**：90 分钟
> **形式**：闭卷，可携带一页 A4 手写公式
> **范围**：L05-L09 + derive_04-06

---

## 第一部分：概念题（每题 5 分，共 30 分）

### Q1
区分 **classifier guidance (CG)** 与 **classifier-free guidance (CFG)**：
- 各需要训练几个网络？
- 哪个工程更易实现？
- 两者数学上等价吗？

---

### Q2
DDIM 与 DDPM 都用同一个训好的网络。它们的本质区别在哪里？为什么这"解耦"如此重要？

---

### Q3
解释 probability flow ODE 的三个关键性质：
- 边缘分布与原 SDE 的关系
- 是否确定性
- 是否可逆

---

### Q4
Stable Diffusion 中"scaling factor = 0.18215"是怎么来的？为什么需要它？

---

### Q5
为什么 SD 的 VAE 用极小的 KL 权重（$\lambda \approx 10^{-6}$）而不是标准 VAE 的 KL 权重？

---

### Q6
判断对错并简要说明：
> "SD 推理时每步要做两次 UNet 前向，所以速度是无 CFG 的一半。"

---

## 第二部分：推导题（每题 10 分，共 40 分）

### Q7
**Anderson 反向 SDE 公式**：

对正向 SDE $dx = f(x, t) dt + g(t) dW$，写出反向 SDE 的形式（包含 score 项）。

简述推导思路（不必完整推导）：从 Fokker-Planck 方程出发，要求反向过程的密度演化与正向匹配。

---

### Q8
**DDIM 推导**：

从两个条件
1. $q_\sigma(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$（与 DDPM 同）
2. $q_\sigma(x_{t-1} | x_t, x_0) = \mathcal{N}(a_t x_0 + b_t x_t, \sigma_t^2 I)$（线性形式）

推出 $a_t, b_t$ 的表达式（用 $\sigma_t$ 与 $\bar\alpha$ 系列表示）。

---

### Q9
**CFG 公式推导**：

从贝叶斯分解 $\nabla_x \log p(x|y) = \nabla_x \log p(y|x) + \nabla_x \log p(x)$ 出发，推出 CFG 公式：
$$\hat\epsilon = \epsilon_\theta(x, t, \emptyset) + s \cdot (\epsilon_\theta(x, t, y) - \epsilon_\theta(x, t, \emptyset))$$

要求显式写出"用 $\nabla \log p(x|y) - \nabla \log p(x)$ 替代 $\nabla \log p(y|x)$"这一关键步骤。

---

### Q10
**Probability flow ODE 推导**：

给定 SDE $dx = f(x, t) dt + g(t) dW$（对应密度 $p_t$），证明 ODE
$$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$$
的解在每个时间 $t$ 与原 SDE 的边缘分布相同。

提示：用 FPE 形式，对比两个 transport equation。

---

## 第三部分：实验设计题（每题 15 分，共 30 分）

### Q11

你在用 SD 1.5 生成图像，prompt 是 "a cat sitting on a chair, photorealistic, 4k"，但生成出来的图：
- 物体识别正确（确实是猫和椅子）
- 但**整体颜色过分饱和、构图过于典型**

(a) 这是什么参数设置导致的？
(b) 如何调整？给出**两个**具体方案，并说明动机
(c) 如果你不能改 CFG，能否用 negative prompt 缓解？怎么写？

---

### Q12

你的 SD 推理在 RTX 3090（24GB）上跑得很慢——512×512 单图 30 秒。

(a) 列出**至少 4 个**可能的优化方向，按预期收益从高到低
(b) 对每个方向给出**具体的代码层动作**（diffusers 怎么调）
(c) 如果你最终把单图时间压到 5 秒，做了哪些权衡？

---

## 提交格式

- 推导题手写或 LaTeX 均可
- 实验设计题用文字回答
- 每题独立编号

---

> 通过 Quiz 2 是进入 P2 部分（DiT / Flow Matching / Video / Embodied AI）的标志。
