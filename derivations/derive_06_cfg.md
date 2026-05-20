# Derivation 06: Classifier-Free Guidance 的贝叶斯推导

> 本手稿对应 L08 内容，详细推导：
> - Classifier guidance 的 score 形式
> - CFG 与 classifier guidance 的精确等价
> - Guidance scale 的几何意义

---

## §1 设置

条件生成的目标：从 $p(x | y)$ 采样。

- $x$：要生成的对象（图像）
- $y$：条件（类别、文本、动作等）

如果我们能拿到 $\nabla_x \log p(x | y)$（条件 score），就可以用 reverse SDE / probability flow ODE 采样。

---

## §2 Classifier Guidance（CG）

### 2.1 贝叶斯分解

由贝叶斯：
$$p(x | y) = \frac{p(x, y)}{p(y)} = \frac{p(y | x) p(x)}{p(y)}$$

两边取对数：
$$\log p(x | y) = \log p(y | x) + \log p(x) - \log p(y)$$

对 $x$ 求梯度（$\log p(y)$ 不含 $x$）：

$$\boxed{\nabla_x \log p(x | y) = \nabla_x \log p(x) + \nabla_x \log p(y | x)}$$

—— **条件 score = 无条件 score + 分类器的对数梯度**。

---

### 2.2 时间相关版本（用于扩散过程）

扩散过程中每个时间步 $t$ 有边缘 $p_t(x_t)$，对应的条件版本是 $p_t(x_t | y)$。

类似分解：
$$\nabla_{x_t} \log p_t(x_t | y) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(y | x_t)$$

含义：
- $\nabla_{x_t} \log p_t(x_t)$：无条件扩散模型的 score（普通 DDPM 训练）
- $\nabla_{x_t} \log p_t(y | x_t)$：**带噪声水平**的分类器（要专门训）

---

### 2.3 分类器要"带噪声水平"

注意 $p_t(y | x_t)$ 中的 $x_t$ 是**带噪**的：
$$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$$

所以**普通的 ImageNet 预训练分类器不能直接用**（它们在干净图像上训练）。

必须训练一个**专门的、能处理任意噪声水平的分类器** $p_\phi(y | x_t, t)$。

---

### 2.4 加 scale 的 CG

实际操作中，引入 scaling factor $w$ 增强 guidance：

$$\hat s(x_t, t, y) = \nabla_{x_t} \log p_t(x_t) + w \cdot \nabla_{x_t} \log p_\phi(y | x_t, t)$$

- $w = 0$：无 guidance（无条件）
- $w = 1$：精确的条件 score（等价于从 $p(x | y)$ 采样）
- $w > 1$：**超条件化**（隐式从 $p(x | y) \cdot p(y | x)^{w-1}$ 采样）

---

### 2.5 噪声预测形式

由 derive_04 §6：$\nabla_{x_t} \log p_t = -\epsilon/\sqrt{1-\bar\alpha_t}$。

所以 CG 写成噪声形式：
$$\hat\epsilon(x_t, t, y) = \epsilon_\theta(x_t, t) - w \sqrt{1-\bar\alpha_t} \cdot \nabla_{x_t} \log p_\phi(y | x_t, t)$$

（注意符号：$\hat\epsilon$ 是"修正后"的预测噪声）

---

## §3 Classifier-Free Guidance（CFG）

### 3.1 关键洞察

**$p(y | x)$ 不一定要显式建模**，只要能间接得到它的梯度就行。

由 §2.1 的等式反过来：
$$\nabla_x \log p(y | x) = \nabla_x \log p(x | y) - \nabla_x \log p(x)$$

—— **隐式分类器**！其梯度就是"条件 score 减无条件 score"。

---

### 3.2 替换 CG 公式

把 §3.1 代入 CG：
$$\hat s(x_t, t, y) = \nabla_{x_t} \log p_t(x_t) + w \cdot \left[\nabla_{x_t} \log p_t(x_t | y) - \nabla_{x_t} \log p_t(x_t)\right]$$

$$= (1 - w) \nabla_{x_t} \log p_t(x_t) + w \cdot \nabla_{x_t} \log p_t(x_t | y)$$

或等价地：
$$\hat s(x_t, t, y) = \nabla_{x_t} \log p_t(x_t) + w \cdot \left[\nabla_{x_t} \log p_t(x_t | y) - \nabla_{x_t} \log p_t(x_t)\right]$$

---

### 3.3 噪声形式（实用）

设 $\epsilon_{\text{cond}} = \epsilon_\theta(x_t, t, y)$，$\epsilon_{\text{uncond}} = \epsilon_\theta(x_t, t, \emptyset)$。

由 noise-score 对应：
$$\hat\epsilon = (1-w) \epsilon_{\text{uncond}} + w \epsilon_{\text{cond}}$$

或者引入 $s = w$（guidance scale）：

$$\boxed{\hat\epsilon = \epsilon_{\text{uncond}} + s \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})}$$

**等价写法**：
$$\hat\epsilon = (1+s') \epsilon_{\text{cond}} - s' \epsilon_{\text{uncond}}, \quad s' = s - 1$$

—— 这就是 CFG 公式。

---

### 3.4 与 CG 严格等价（数学上）

回顾：
- CG: $\hat s = \nabla \log p + w \nabla \log p_\phi(y|x)$（需要 $\phi$）
- CFG: $\hat s = \nabla \log p + w (\nabla \log p^{\text{cond}} - \nabla \log p)$（不需要 $\phi$）

只要 $\nabla \log p^{\text{cond}}$ 准确（用同一网络的条件分支预测），两者**精确等价**。

但 CFG 有两个优势：
1. 不需要训独立分类器
2. "隐式分类器" $\nabla \log p^{\text{cond}} - \nabla \log p$ 可能比独立训练的 $\nabla \log p_\phi$ 更准（因为它直接利用了 $p$ 中"条件信息"的内在表示）

---

## §4 训练实现：Conditional Dropout

### 4.1 算法

让**同一个网络** $\epsilon_\theta(x_t, t, y)$ 兼任条件与无条件预测：

```python
def cfg_training_step(model, x0, y, t, schedule, p_uncond=0.1):
    B = x0.shape[0]
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, schedule, noise)

    # 随机将一部分样本的条件替换为 ∅ (null)
    mask = torch.rand(B, device=x0.device) < p_uncond
    y_input = y.clone()
    y_input[mask] = NULL_TOKEN  # 或全零向量

    pred = model(x_t, t, y_input)
    return F.mse_loss(pred, noise)
```

**关键细节**：
- $p_{\text{uncond}}$ 取 0.1-0.2（论文 0.1，SD 0.1）
- 空条件 $\emptyset$ 在 text-to-image 中是空字符串 ""，编码后得到固定的"empty text embedding"
- 也可以用专门的 "null embedding"（一个可学的向量）

---

### 4.2 为什么 0.1 而不是 0.5？

经验最优值 0.1-0.2。背后直觉：
- 太低（<0.05）：网络对空条件输入欠采样，无条件分支学不好
- 太高（>0.3）：网络对条件依赖度不够，CFG 效果减弱

**消融实验**（Imagen / CFG 论文）：
| $p_{\text{uncond}}$ | FID @ best CFG scale |
|---------------------|----------------------|
| 0.05 | 4.5 |
| 0.10 | **3.9** |
| 0.20 | 4.1 |
| 0.50 | 5.8 |

---

## §5 推理实现：Batch Concat 优化

朴素实现要两次前向：
```python
eps_cond = model(x, t, y)        # forward pass 1
eps_uncond = model(x, t, null)   # forward pass 2
eps = eps_uncond + s * (eps_cond - eps_uncond)
```

优化：**把两个输入拼成 batch**，一次前向：

```python
x_in = torch.cat([x, x], dim=0)        # (2B, ...)
y_in = torch.cat([null_y, y], dim=0)   # (2B, ...)
t_in = t.repeat(2)
out = model(x_in, t_in, y_in)
eps_uncond, eps_cond = out.chunk(2)
eps = eps_uncond + s * (eps_cond - eps_uncond)
```

显存翻倍，但充分利用 GPU 并行，速度上几乎和单次前向相同。

> **🔑 这是 SD 推理的标准实现**。

---

## §6 Guidance Scale 的几何意义

### 6.1 向量解释

把 score 想成"指向高密度方向"的向量场。

```
                           （高密度区，类别 y 的数据）
                                    *
                                   /
                                  /  ← ε_cond
                                 /     （条件 score）
                                /
    x_t   *──────────────────/
            \
             \   ← ε_uncond
              \    （无条件 score）
               \
                * (高密度区，所有类别数据)
```

CFG 的 $s \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$ 是从 $\epsilon_{\text{uncond}}$ 到 $\epsilon_{\text{cond}}$ 这段向量。

- $s = 0$：只用 $\epsilon_{\text{uncond}}$（无引导）
- $s = 1$：到达 $\epsilon_{\text{cond}}$（精确条件采样）
- $s > 1$：**外推**——超过条件 score，更强"听话"

---

### 6.2 为什么大 $s$ 会出问题？

大 $s$ 意味着 $\hat\epsilon$ 远超出训练分布的范围。神经网络在这种 OOD 输入下行为未知，常见症状：
- 颜色过饱和
- 构图过分典型化
- 严重 artifact

---

### 6.3 Imagen 的 Dynamic Thresholding

为缓解大 $s$ 的问题，Imagen 提出 **dynamic thresholding**：

在采样过程中，把 $\hat x_0$（用 $\hat\epsilon$ 反推得到）clip 到合理范围。

```python
x_hat_0 = (x_t - sqrt(1-a_bar) * eps_hat) / sqrt(a_bar)
# Dynamic thresholding
percentile = 99.5
threshold = max(1.0, x_hat_0.abs().quantile(percentile / 100.0))
x_hat_0 = x_hat_0.clamp(-threshold, threshold) / threshold
```

效果：在 $s = 10-20$ 下仍能保持质量。

---

## §7 Negative Prompt 的本质

SD 用户常用 negative prompt：
$$\hat\epsilon = \epsilon_\theta(x, t, y_{\text{neg}}) + s \cdot \left[\epsilon_\theta(x, t, y_{\text{pos}}) - \epsilon_\theta(x, t, y_{\text{neg}})\right]$$

—— **把 $\emptyset$ 换成 $y_{\text{neg}}$**。

数学含义：
- 不只是"放大 $\epsilon_{\text{cond}}$ 方向"
- 而是"放大从 $y_{\text{neg}}$ 到 $y_{\text{pos}}$ 的方向"

效果：远离 $y_{\text{neg}}$ 描述的特征（伪影、低质量、特定风格）。

---

## §8 CFG 的训练-推理一致性

### 8.1 形式上的"奇怪"

训练时 $p_{\text{uncond}} = 0.1$，意味着网络见过 ~10% 的空条件输入。推理时 CFG 公式做了**外推**（$s > 1$），相当于走向网络从未见过的输入分布。

为什么能 work？

**答案**：CFG 不是改变网络输出本身，而是**把两个有效输出做线性外推**。网络在 $\emptyset$ 和 $y$ 两个输入下都给出合理的 $\epsilon$ 预测，它们的"差向量"是有意义的方向。

---

### 8.2 训练时不用 CFG

注意：**训练时 loss 中不出现 CFG 公式**！只是在推理时用 CFG 外推。

这种"训练时简单、推理时玩花样"的设计是 CFG 优雅的地方。

---

## §9 进阶讨论：CFG 的偏差

### 9.1 隐式分布偏差

CFG 在 $s > 1$ 时**不是**从 $p(x | y)$ 采样，而是从一个偏差分布：
$$\tilde p(x | y) \propto p(x | y) \cdot \left(\frac{p(x | y)}{p(x)}\right)^{s-1}$$

这个分布"更窄"——更多 typical $y$ 样本，更少 atypical 样本。

**好处**：质量提升、prompt fidelity 强。
**坏处**：多样性下降，可能 mode collapse。

---

### 9.2 研究方向

- **Adaptive CFG**：让 $s$ 随 $t$ 变化
- **CFG distillation**（Meng 2023）：训学生网络直接预测 $\hat\epsilon$，单次前向
- **CFG-free**：让条件模型自学到 strong guidance

---

## §10 自查题

1. 从 $\log p(x|y)$ 的贝叶斯分解推出 CG 公式
2. 解释为什么 $\nabla \log p(y|x) = \nabla \log p(x|y) - \nabla \log p(x)$
3. 推出 CFG 噪声形式的两种等价表达
4. 解释 conditional dropout 训练为什么有效

---

## §11 参考文献

- Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis*, NeurIPS 2021（CG 提出）
- Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS workshop 2021（CFG 提出）
- Saharia et al., *Imagen*, NeurIPS 2022（dynamic thresholding）
- Meng et al., *On Distillation of Guided Diffusion Models*, CVPR 2023
