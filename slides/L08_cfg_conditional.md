# Lecture 08｜Classifier-Free Guidance 与条件生成

> **本讲目标**
> - 理解条件生成的两种范式：classifier guidance 与 classifier-free guidance
> - 完整推导 CFG 公式
> - 理解 guidance scale 的物理意义与质量-多样性 trade-off
> - 掌握 cross-attention 注入条件的工程细节
> - 为 W8（Stable Diffusion）做准备

> **对应论文**：
> - Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis*, NeurIPS 2021（classifier guidance）
> - Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS workshop 2021（CFG）

---

## §1 无条件 vs 条件生成

### 1.1 无条件生成（前 7 讲）

到目前为止学的 DDPM 都是无条件的：
$$p_\theta(x_0) \approx p_{\text{data}}(x_0)$$

**问题**：随机生成，无法控制内容。

### 1.2 条件生成

引入条件 $y$（标签、文本、图像、动作等）：
$$p_\theta(x_0 | y) \approx p_{\text{data}}(x_0 | y)$$

**应用**：
- 文本 → 图像（SD, Imagen）
- 图像 → 图像（img2img, inpainting）
- 类别 → 图像（class-conditional generation）
- 文本 → 视频/动作（VideoLDM, Diffusion Policy）

---

## §2 朴素方法：直接训练条件模型

最直接做法：把条件 $y$ 作为额外输入，训练 $\epsilon_\theta(x_t, t, y)$。

```python
def p_losses_conditional(model, x0, y, t, schedule):
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, schedule, noise)
    pred = model(x_t, t, y)  # y 是条件
    return F.mse_loss(pred, noise)
```

**问题**：模型可能"忽略"条件（学习捷径），生成质量虽然好，但 condition fidelity 差。

> **🔑 这就是为什么需要 guidance**：强制模型"听话"。

---

## §3 Classifier Guidance（CG）

### 3.1 核心思想

利用贝叶斯：
$$\nabla_x \log p(x | y) = \nabla_x \log p(x) + \nabla_x \log p(y | x)$$

其中：
- $\nabla_x \log p(x)$：无条件 score（普通 DDPM）
- $\nabla_x \log p(y | x)$：分类器对 $x$ 给出"$y$ 类"概率的梯度

**思路**：训一个独立的分类器 $p_\phi(y | x)$，把它的梯度加到 score 上。

---

### 3.2 加强版：带 scale 的 CG

实际操作中，加一个 scaling factor $w$ 控制 guidance 强度：

$$\hat\epsilon_\theta(x_t, t, y) = \epsilon_\theta(x_t, t) - w \cdot \sqrt{1-\bar\alpha_t} \cdot \nabla_x \log p_\phi(y | x_t)$$

- $w = 0$：无条件
- $w = 1$：等价于条件分布
- $w > 1$：放大条件信息（"超条件化"）

**Dhariwal & Nichol 发现**：$w > 1$ 时质量显著提升（但多样性下降）。

---

### 3.3 CG 的代价

要训练一个**专门的、能处理带噪图像的分类器** $p_\phi(y | x_t, t)$：
- 不能用 ImageNet 预训练分类器（它们在干净图像上训的）
- 要单独训练，工作量大
- 分类器的质量直接影响 guidance 效果

> **🔑 这是 CFG 要解决的核心问题**：能否不要额外的分类器？

---

## §4 Classifier-Free Guidance（CFG）

### 4.1 核心想法

Ho & Salimans 的精妙发现：**用同一个网络兼任条件与无条件预测**。

训练时：
- 大部分 batch 用条件 $y$（如 90%）
- 小部分 batch 用"空"条件 $\emptyset$（如 10%）
- 网络同时学到 $\epsilon_\theta(x_t, t, y)$ 和 $\epsilon_\theta(x_t, t, \emptyset)$

推理时，把两者**线性组合**（外推）：

$$\boxed{\hat\epsilon_\theta(x_t, t, y) = (1 + w) \cdot \epsilon_\theta(x_t, t, y) - w \cdot \epsilon_\theta(x_t, t, \emptyset)}$$

或等价写法（$s = w + 1$）：
$$\hat\epsilon = \epsilon_{\text{uncond}} + s \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$$

- $s = 0$：无条件
- $s = 1$：条件
- $s > 1$：放大条件方向

---

### 4.2 训练算法

```python
def p_losses_cfg(model, x0, y, t, schedule, p_uncond=0.1):
    # 随机将一些样本的条件替换为 ∅
    B = x0.shape[0]
    mask = torch.rand(B) < p_uncond  # 10% 概率
    y_input = y.clone()
    y_input[mask] = NULL_TOKEN  # 或全零向量

    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, schedule, noise)
    pred = model(x_t, t, y_input)
    return F.mse_loss(pred, noise)
```

**关键点**：
- $p_{\text{uncond}}$ 取 10-20%（论文用 0.1，SD 用 0.1）
- 空条件 $\emptyset$ 在 text 中通常是空 string ""
- 也可以用专门的"null embedding"

---

### 4.3 推理算法

```python
def sample_cfg(model, schedule, y, w=7.5, ...):
    """w 是 guidance scale，SD 默认 7.5"""
    x = torch.randn(...)
    for t in reversed(range(T)):
        # 同时跑条件和无条件预测（同一个网络）
        eps_cond = model(x, t, y)
        eps_uncond = model(x, t, NULL)

        # CFG 线性外推
        eps = eps_uncond + w * (eps_cond - eps_uncond)

        # 用 eps 做正常采样
        x = p_sample_step(x, t, eps)
    return x
```

**计算代价**：每步要做**两次**网络前向（条件 + 无条件）。这是 CFG 的主要开销。

**工程优化**：把条件和无条件 batch 一起送 GPU（batch concat）：
```python
x_in = torch.cat([x, x], dim=0)              # (2B, ...)
y_in = torch.cat([y, null_y], dim=0)         # (2B, ...)
out = model(x_in, t.repeat(2), y_in)         # 一次前向
eps_cond, eps_uncond = out.chunk(2, dim=0)
```

---

## §5 CFG 公式的数学解释

### 5.1 等价于隐式分类器

CFG 形式上看起来"凭空"，实际上有严格数学含义。

由贝叶斯：
$$\log p(y | x) = \log p(x | y) - \log p(x) + \text{const}$$

对 $x$ 求梯度：
$$\nabla_x \log p(y | x) = \nabla_x \log p(x | y) - \nabla_x \log p(x)$$

—— **隐式分类器**！它的梯度就是"条件 score 减无条件 score"。

代入 classifier guidance 公式，得到 CFG。所以 **CFG ≡ classifier guidance**，只是用网络的两次前向"自带分类器"。

---

### 5.2 几何直觉

把 score $\nabla_x \log p$ 想成"指向高密度方向"的向量场。

```
              (高密度区，类别 y 数据)
                       *
                      /
                     /  ← 条件 score
                    /
   x_t   *---------/
           \
            \     ← 无条件 score
             \
              *  (高密度区，所有类数据)
```

CFG 的 $s \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$ 是这两个方向的**差分**，乘 $s$ 后**强化条件方向**。

---

## §6 Guidance Scale 的影响

### 6.1 质量-多样性 trade-off

CFG 是 Stable Diffusion 用户最熟悉的旋钮。

| Guidance Scale ($s$) | 现象 |
|----------------------|------|
| 1.0 | 等价于纯条件采样，质量平庸，多样性高 |
| 3.0 | 略改进，更符合 prompt |
| 7.5 (SD 默认) | 质量和 prompt fidelity 良好平衡 |
| 10-15 | 过强引导，颜色过饱和，构图更典型 |
| 20+ | 严重伪影，"扭曲" |

### 6.2 为什么大 scale 会有伪影？

直觉：$s$ 大时，$\hat\epsilon$ 远超出训练分布中观察到的范围。模型在外推区域行为未知，产生 artifact。

**缓解方法**：
- Dynamic thresholding（Imagen）：把预测的 $\hat x_0$ clip 到合理范围
- Rescaling CFG（SD 2.1+）：保持 $\|\hat\epsilon\| \approx \|\epsilon\|$

---

## §7 条件注入的工程方式

CFG 只解决"如何利用条件"，还需要决定"如何把条件信息送入网络"。

### 7.1 类别条件（class-conditional）

类别用 embedding lookup：
```python
self.class_emb = nn.Embedding(num_classes + 1, time_dim)  # +1 for null
# forward 时：
y_emb = self.class_emb(y)
t_emb = self.time_emb(t)
combined = t_emb + y_emb  # 加法注入
```

注入方式与 time embedding 相同（broadcasting 加到每个 ResBlock）。

---

### 7.2 文本条件（text-to-image）

文本 → token sequence → text encoder（CLIP, T5）→ token embeddings

注入方式：**Cross-Attention**

```python
class CrossAttention(nn.Module):
    """
    Q from image features (B, C_img, H, W) → (B, HW, C_img)
    K, V from text embeddings (B, T_text, C_text)
    """
    def __init__(self, query_dim, context_dim, heads=8):
        super().__init__()
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.heads = heads

    def forward(self, x, context):
        # x: image features (B, HW, C_img)
        # context: text embeddings (B, T_text, C_text)
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        # multi-head attention as usual
        ...
```

每个 ResBlock 后面插入 cross-attention 层，让图像特征"看"文本。

---

### 7.3 图像条件（ControlNet 等）

更复杂——图像条件与目标图像形状相同，需要"对应位置注入"。
- **ControlNet**（Zhang 2023）：复制 U-Net 编码器的副本接受条件图，输出 feature 加到主网络
- **T2I-Adapter**：更轻量的 adapter 方式

> **🔑 这部分 W9 详细展开。**

---

## §8 Negative Prompt 的本质

SD 用户都用过 negative prompt（"low quality, blurry"）。它的数学含义？

**Negative prompt 是把 $\emptyset$ 替换为 $y_{\text{neg}}$**：

$$\hat\epsilon = \epsilon_\theta(x, t, y_{\text{neg}}) + s \cdot (\epsilon_\theta(x, t, y_{\text{pos}}) - \epsilon_\theta(x, t, y_{\text{neg}}))$$

不再是"放大条件方向"，而是"放大从 $y_{\text{neg}}$ 到 $y_{\text{pos}}$ 的方向"。

**直觉**：告诉模型"不仅要像 $y_{\text{pos}}$，还要远离 $y_{\text{neg}}$"。

效果：可以减少特定不想要的特征（伪影、低质量、特定风格）。

---

## §9 CFG 的局限

### 9.1 计算开销

每步两次网络前向，推理成本翻倍。这是为什么 SD 不能更快的根本原因之一。

**研究方向**：
- **Guidance distillation**（Meng 2023）：训一个学生网络直接学外推后的 score，单次前向
- **CFG-free**：让条件模型自己学到 strong guidance

---

### 9.2 静态 scale 不一定最优

固定 $s$ 整个采样过程不是最优——某些时间步需要更强 guidance，某些需要更弱。

**研究方向**：
- **CFG schedule**：让 $s$ 随 $t$ 变化
- **Adaptive CFG**

---

## §10 本讲核心要点

1. **条件生成需要 guidance**：单纯训条件模型不够"听话"
2. **Classifier Guidance** 用独立分类器的梯度，代价是要训分类器
3. **CFG** 用同一网络同时学条件与无条件，**推理时两次前向 + 线性外推**
4. **CFG 公式**：$\hat\epsilon = \epsilon_{\text{uncond}} + s \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$
5. **CFG ≡ classifier guidance**（用隐式分类器）
6. **Guidance scale** 是质量-多样性 trade-off 的旋钮，SD 默认 7.5
7. **Cross-Attention** 是文本条件注入的标准方式
8. **Negative prompt** 本质是把无条件换成负条件

---

## §11 课后任务

### 必做

1. **公式推导**：从 $\log p(y | x) = \log p(x | y) - \log p(x) + \text{const}$ 出发，推出 CFG 的 score 等价形式。

2. **代码任务**：在 Project 1 的 DDPM 基础上添加类别条件（MNIST 数字 0-9）
   - 修改 U-Net 接受 `class_label`
   - 实现 CFG 训练（10% dropout）
   - 实现 CFG 推理（scale=3.0）
   - 比较不同 scale 下生成的数字质量
   - 提交 `class_cond_ddpm/` 子目录

3. **Reading note**：阅读 CFG 论文，重点理解"为什么不需要单独分类器"。

### 选做

4. **进阶**：实现 negative prompt 机制，对一个预训练的 class-conditional DDPM 做"远离类别 0"的采样。

5. **挑战**：实现 cross-attention 用于 MNIST 数字"文本"条件（"the digit five"）。需要小型 text encoder。

---

## §12 推荐进一步阅读

| 资源 | 重点 |
|------|------|
| CFG 论文（Ho & Salimans 2021） | 简短必读 |
| Dhariwal & Nichol 2021 | classifier guidance + Diffusion Beat GAN |
| Imagen 论文（Saharia 2022） | dynamic thresholding |
| 推导手稿 `derive_06_cfg.pdf` | 完整数学推导 |

---

## §13 常见问题

**Q: SD 里 CFG scale 设为 7.5 是怎么定下来的？**

A: 经验值。LDM 论文实验显示 7-10 是 sweet spot；SD 默认 7.5 是工程上的折中选择。不同模型、不同 prompt 的最优 scale 可能不同——这是为什么有 prompt engineer。

**Q: CFG 训练时 10% dropout 的具体实现细节？**

A: 对 text-to-image：把 caption 替换为空字符串 ""。对 class-conditional：用专门的 "null class" token (id = num_classes)。注意空 prompt 不是"删掉条件"，而是给个明确的"无条件"信号。

**Q: 为什么 CFG 公式形式是"外推"（extrapolation）而非"插值"？**

A: $w + 1 > 1$ 意味着我们走出"条件 score - 无条件 score" 这段向量的终点。**外推强化条件方向**，符合"超条件化"的物理需求。如果用 $w \in [0, 1]$ 插值，相当于"软化条件"，与 guidance 的目的相反。

**Q: 训练时 conditional dropout 用 0.5 比 0.1 好吗？**

A: 不好。论文实验显示 0.1-0.2 最优。太高（0.5）让网络对条件依赖度不够；太低（<0.05）让无条件分支学不好。

**Q: 我能在已训好的条件 DDPM 上加 CFG 吗？**

A: 不能直接加。CFG 要求训练时见过空条件输入——否则推理时网络对空输入的反应是 OOD（out of distribution）的。需要在 fine-tuning 中补 conditional dropout 训练。

---

> **下一讲预告**：L09 是本课程的"应用高峰"——**Latent Diffusion Model 与 Stable Diffusion**。我们将看到 LDM 如何把 diffusion 从像素空间搬到 latent 空间，使 512×512 生成成为可能。同时也是 SD 工程的完整解剖。
