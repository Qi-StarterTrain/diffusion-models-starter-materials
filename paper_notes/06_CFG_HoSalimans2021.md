# 论文导读 06: CFG (Ho & Salimans, NeurIPS Workshop 2021)

**标题**：Classifier-Free Diffusion Guidance
**作者**：Jonathan Ho, Tim Salimans (Google Brain)
**核心地位**：⭐⭐⭐⭐⭐ 现代条件扩散的"无形之手"。SD、Imagen、Sora 都用 CFG

---

## 一、为什么必读

- **最重要的工业级 trick**：每个用过 SD 的人都用过 CFG（默认 scale=7.5）
- 工程极其简洁：训练加 10% conditional dropout，推理两次前向
- **数学等价于 classifier guidance**，但不需要训分类器

这是一篇短论文（只有 8 页），但影响力远超长度。**1-2 小时通读+推导**就够。

---

## 二、阅读路线图

| 章节 | 优先级 |
|------|--------|
| §1 Intro | ✅ |
| §2 Background | 跳过 |
| **§3 Classifier-free guidance** | 🔴 核心 |
| §4 Experiments | ✅ |

---

## 三、核心公式

### 训练
$$\mathcal{L} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t, t, y)\|^2\right]$$
其中 $y$ 以概率 $p_{\text{uncond}} = 0.1$ 被替换为 $\emptyset$（null token）。

### 推理
$$\hat\epsilon = \epsilon_\theta(x_t, t, \emptyset) + s \cdot \left[\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset)\right]$$

---

## 四、关键洞察

1. **隐式分类器**：$\nabla \log p(y|x) = \nabla \log p(x|y) - \nabla \log p(x)$ —— 用扩散模型的两次前向"得到"分类器梯度
2. **训练时不出现 CFG 公式**：只在推理时线性外推
3. **外推（$s > 1$）超出训练分布但 work**：这是 CFG "魔法"的来源

---

## 五、与 CG 的数学等价（必看）

由贝叶斯：
$$\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)$$

CG 用独立分类器估计 $\nabla \log p(y|x)$。
CFG 用 $\nabla \log p(x|y) - \nabla \log p(x)$ 替代 $\nabla \log p(y|x)$。

两者数学等价，但 CFG 的"分类器"更准（因为直接来自联合训练的模型）。

---

## 六、容易误读

### 1. "CFG 一定要 $p_{\text{uncond}}=0.1$？"

实验最优 0.1-0.2。**太低（<0.05）**：网络对空条件输入欠采样。**太高（>0.3）**：网络对条件依赖不足。

---

### 2. "Negative prompt 是 CFG 的扩展？"

是。CFG 公式中把 $\emptyset$ 换成 $y_{\text{neg}}$：
$$\hat\epsilon = \epsilon_\theta(y_{\text{neg}}) + s (\epsilon_\theta(y_{\text{pos}}) - \epsilon_\theta(y_{\text{neg}}))$$

**直觉**："远离" $y_{\text{neg}}$，"靠近" $y_{\text{pos}}$。

---

### 3. "CFG 后采样的是 $p(x|y)$？"

**严格说不是**。$s > 1$ 时实际采样的是：
$$\tilde p(x|y) \propto p(x|y) \cdot (p(x|y)/p(x))^{s-1}$$

—— 一个"偏向典型 $y$ 样本"的偏差分布。这就是 CFG 牺牲多样性换质量的来源。

---

## 七、与其他论文的关系

- **CG (Dhariwal 2021)**：直接前作。CFG 是 CG 的简化
- **Imagen (Saharia 2022)**：用 CFG + dynamic thresholding 解决大 $s$ 伪影
- **SD (Rombach 2022)**：把 CFG 工程化到 LDM
- **DALL-E 2 (Ramesh 2022)**：也用 CFG

几乎所有 SOTA text-to-image 模型都用 CFG。

---

## 八、工程实战

### Batch concat 优化
```python
x_in = torch.cat([x, x], dim=0)
y_in = torch.cat([null_y, y], dim=0)
out = model(x_in, t.repeat(2), y_in)
eps_uncond, eps_cond = out.chunk(2)
eps = eps_uncond + s * (eps_cond - eps_uncond)
```

显存翻倍，但 GPU 并行让速度几乎不变。

### CFG distillation（进阶）
Meng 2023 提出训一个学生网络直接学外推后的 score，让推理只需 1 次前向。

---

## 九、思考题

1. 在哪些情况下 CFG **不应**用（除推理外）？训练时直接用 CFG 公式行吗？
2. CFG 的"隐式分类器" $\nabla \log p(x|y) - \nabla \log p(x)$ 在某些 $x$ 上可能不可靠。哪些情况？
3. 设计一个实验，量化 $s$ 与多样性的 trade-off（用 recall 指标）

---

## 十、引用

```
@article{ho2021classifierfree,
  title={Classifier-Free Diffusion Guidance},
  author={Ho, Jonathan and Salimans, Tim},
  journal={NeurIPS Workshop on DGM and Downstream Applications},
  year={2021}
}
```
