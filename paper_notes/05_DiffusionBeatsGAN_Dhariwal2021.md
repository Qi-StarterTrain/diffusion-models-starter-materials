# 论文导读 05: Diffusion Models Beat GANs (Dhariwal & Nichol, NeurIPS 2021)

**标题**：Diffusion Models Beat GANs on Image Synthesis
**作者**：Prafulla Dhariwal, Alex Nichol (OpenAI)
**核心地位**：⭐⭐⭐⭐ 历史时刻论文。首次在 256×256 ImageNet 上击败 BigGAN

---

## 一、为什么必读

- 提出 **classifier guidance**：扩散模型条件生成的第一种工程化方法
- 系统性 ablation：哪些 trick 影响 FID 多少（必读）
- 在 256×256/512×512 ImageNet 上的 sample 至今仍是经典展示

虽然 CFG 后来取代了 classifier guidance，但**这篇是 CFG 的概念前作**——不读 CG 就不懂 CFG 的动机。

---

## 二、阅读路线图

| 章节 | 优先级 |
|------|--------|
| §1 Intro | ✅ |
| §2 Background | 跳过 |
| **§3 Architecture Improvements** | 🔴 核心 |
| **§4 Classifier Guidance** | 🔴 核心 |
| §5 Results | ✅ |
| §6 Limitations | ✅ |

---

## 三、§3 架构改进（系统 ablation）

论文系统性地探索"什么让 DDPM 更好"：

| 改进 | FID 改善 |
|------|---------|
| 增加宽度 | -0.3 |
| 增加深度 | -0.2 |
| **更多 attention 分辨率**（8/16/32 都加） | **-1.0** |
| **BigGAN-style residual**（rescaling） | -0.3 |
| **Adaptive Group Norm**（注入 timestep） | -0.4 |

**核心发现**：attention 在更多分辨率上加，比单纯加宽 + 深效果好。这是后续 SD/Imagen 的设计依据。

---

## 四、§4 Classifier Guidance（必读）

### 4.1 核心公式
$$\hat\epsilon = \epsilon_\theta(x_t, t) - w \sqrt{1-\bar\alpha_t} \cdot \nabla_{x_t} \log p_\phi(y | x_t, t)$$

### 4.2 工程关键

1. 需要训一个**专用分类器** $p_\phi(y|x_t, t)$，能处理任意噪声水平的输入
2. 训练分类器在所有 $t$ 的样本上（不只是干净图）
3. $w$ 是 guidance scale，论文用 1.0-10.0

### 4.3 实验结果

| 模型 | FID (256 ImageNet) |
|------|---------------------|
| BigGAN-deep | 6.95 |
| Improved DDPM | 12.26 |
| **+ Classifier Guidance (w=1)** | **4.59** |

—— 首次扩散模型在 256×256 ImageNet 上击败 GAN。

---

## 五、容易误读

### 1. "Classifier Guidance 比 CFG 好？"

**已被 CFG 取代**。原因：
- CG 要训独立分类器（工程成本）
- CFG 数学等价但更简洁
- CFG 推广到 text-to-image 自然，CG 难

但**CG 在某些专门场景仍有用**：如 robust 模型（用对抗鲁棒的分类器）、外部信号注入。

---

### 2. "Adaptive Group Norm 是什么？"

文中称 AdaGN：把时间 embedding 通过线性层映射为 GroupNorm 的 $\gamma, \beta$ 参数：
```python
gamma, beta = self.proj(t_emb).chunk(2, dim=-1)
x = (1 + gamma) * self.norm(x) + beta
```

这是 BigGAN 的 AdaBN 在扩散模型上的版本。**SD 仍在用类似机制**。

---

### 3. "BigGAN-style residual" 具体指什么？

每个 ResBlock 内：
```python
def forward(self, x):
    h = conv1(x) + scale * upsample(x)  # rescale skip
    return h
```

`scale = 1/√2` 的目的：保持方差稳定。**SD UNet 用了这个技巧**。

---

## 六、与其他论文的关系

- **Improved DDPM**：同一作者的前作（cosine schedule、learned variance）
- **CFG (Ho & Salimans 2021)**：本论文的直接后续，替代 classifier guidance
- **SDEdit, DiffEdit**：基于 classifier guidance 思想的条件生成方法

---

## 七、思考题

1. 为什么训 robust 分类器（adv-training）会让 classifier guidance 效果变差？（提示：robust 分类器的 logit 梯度更"平"）
2. 实验显示在 ImageNet 上 $w \approx 1$ 就足够，但 SD 用 $w \approx 7.5$。差异从何而来？（提示：文本条件 vs 类别条件，guidance 难度不同）
3. 如果用 CLIP 作为"分类器"（CLIP guidance），相比类别分类器的优势？

---

## 八、引用

```
@inproceedings{dhariwal2021diffusion,
  title={Diffusion Models Beat GANs on Image Synthesis},
  author={Dhariwal, Prafulla and Nichol, Alexander},
  booktitle={NeurIPS},
  year={2021}
}
```
