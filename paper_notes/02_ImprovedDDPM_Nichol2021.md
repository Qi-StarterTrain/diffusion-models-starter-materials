# 论文导读 02: Improved DDPM (Nichol & Dhariwal, ICML 2021)

**标题**：Improved Denoising Diffusion Probabilistic Models
**作者**：Alex Nichol, Prafulla Dhariwal (OpenAI)
**核心地位**：⭐⭐⭐⭐ DDPM 的工程改进，引入 cosine schedule、learned variance、importance sampling

---

## 一、为什么必读

DDPM 在视觉质量上击败了 GAN，但在 NLL 上落后于最佳 likelihood 模型。这篇是修补这个差距的工程典范，三个改进各自都很经典：
1. Cosine schedule（大图标配）
2. Learned variance
3. Importance sampling of $t$

---

## 二、阅读路线图

| 章节 | 是否必读 |
|------|--------|
| §1 Intro | ✅ |
| §2 Background | 跳过（已学 DDPM） |
| **§3 Improving the Log-Likelihood** | 🔴 核心 |
| §3.1 Learning $\Sigma_\theta$ | 🔴 核心 |
| §3.2 Improving the noise schedule | 🔴 核心（cosine） |
| §3.3 Reducing gradient noise | 🔴 核心（importance sampling） |
| §4 Improving Sampling Speed | ✅ 概览即可 |
| §5 Comparing with GANs | 选读 |
| §6 Scaling Properties | ✅ 必读（scaling law 重要洞察） |

---

## 三、核心公式

### Cosine Schedule
$$\bar\alpha_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$$

其中 $s = 0.008$ 是小偏移量。

---

### Learned Variance
$$\Sigma_\theta = \exp(v \log \beta_t + (1-v) \log \tilde\beta_t)$$

网络预测 $v \in [0, 1]$，在 $\log\beta_t$（上界）与 $\log\tilde\beta_t$（下界）之间插值。

---

### Hybrid Loss
$$\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{\text{simple}} + \lambda \mathcal{L}_{\text{vlb}}, \quad \lambda = 0.001$$

注意：**$\mathcal{L}_{\text{vlb}}$ 对 $\epsilon$ 通道 stop-gradient**——只用来训练方差通道。

---

### Importance Sampling
$$p(t) \propto \sqrt{\mathbb{E}[L_t^2]}$$

按 loss 平方均值的开方采样 $t$。

---

## 四、常被误读

### 1. "Cosine schedule 总比 linear 好"

**错**。论文 §3.2 明确：cosine 在 64×64+ 上改进显著，但 CIFAR-10（32×32）上**几乎相同或略差**。

工程实践：先用 linear baseline，再试 cosine 做对比，根据数据选择。

---

### 2. "Learned variance 工业界普及了？"

**没有**。SD 1.x/2.x 都没用 learned variance。原因：
- $\lambda$ 调参敏感（太大破坏 noise prediction）
- FID 提升不显著（仅 NLL 改进）
- 工程上加复杂度但收益不直接

但教学价值仍很高——理解 ELBO 与 simplified loss 的差距。

---

### 3. "Importance sampling 一定改善训练？"

§3.3 说改善"after sufficient training"。**早期训练阶段** loss 噪声大，importance sampling 可能加剧抖动。论文建议：
- 训练前几千步用均匀采样
- 之后切换到 importance sampling

---

## 五、Scaling Properties（§6 必读）

| Params | FID (CIFAR-10) | NLL |
|--------|----------------|-----|
| 21M | 3.21 | 3.69 |
| 35M | 2.94 | 3.62 |
| 51M | 2.83 | 3.57 |
| 137M | 2.45 | - |

**结论**：扩散模型的 FID 与参数量近似 power-law。**意义重大**——这是后续 SD、DiT、Imagen 敢于堆参数的依据。

---

## 六、与其他论文的关系

- DDPM (Ho 2020)：**直接前作**
- Diffusion Beat GANs (Dhariwal 2021)：**直接后作**——Imp DDPM 的作者同一伙人，加上 classifier guidance 后击败 BigGAN
- Score SDE：从连续时间角度统一 cosine schedule（变成 $\beta(t)$ 的不同函数选择）

---

## 七、思考题

1. 为什么 hybrid loss 的 $\lambda$ 只能取 0.001 这么小？尝试 0.1 会怎样？
2. Importance sampling 在 batch 内 $t$ 重复的概率会变大（高 loss 区域被多选）。这是否影响 batch 多样性？
3. Cosine schedule 在 t=0 处的"smooth" 设计（$s=0.008$）为什么不能省？

---

## 八、引用

```
@inproceedings{nichol2021improved,
  title={Improved denoising diffusion probabilistic models},
  author={Nichol, Alexander Quinn and Dhariwal, Prafulla},
  booktitle={ICML},
  year={2021}
}
```
