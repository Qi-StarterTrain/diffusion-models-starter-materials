# 论文导读 01: DDPM (Ho et al., NeurIPS 2020)

**标题**：Denoising Diffusion Probabilistic Models
**作者**：Jonathan Ho, Ajay Jain, Pieter Abbeel
**核心地位**：⭐⭐⭐⭐⭐ 现代扩散模型的奠基论文，**必读全文**

---

## 一、为什么这篇必读

如果整个课程只读一篇论文，就是它。
- 首次让扩散模型在视觉任务上击败 GAN
- 推出的 simplified loss 是后续所有 DDPM 变种的基石
- 论文写作清晰，公式严谨

> 读不懂这篇等于扩散模型没入门。

---

## 二、阅读路线图

| 章节 | 是否必读 | 备注 |
|------|--------|------|
| §1 Intro | ✅ 必读 | 历史动机 |
| §2 Background | ✅ 必读 | 简短回顾 NCSN |
| **§3 Diffusion models and denoising autoencoders** | 🔴 **核心** | 整篇精华 |
| §3.1 Forward process & $L_T$ | 🔴 核心 | 注意 $L_T$ 几乎为 0 的推理 |
| §3.2 Reverse process & $L_{1:T-1}$ | 🔴 核心 | $\epsilon$ 预测参数化 |
| §3.3 Data scaling & $L_0$ | ✅ 必读 | 离散像素 likelihood |
| §3.4 Simplified loss | 🔴 核心 | **simplified loss 出处** |
| §4 Experiments | ✅ 必读 | 看哪些 trick 影响大 |
| §4.1-§4.3 | ✅ 必读 | 主要结果 |
| §A-§B 附录 | 选读 | 完整证明 |

**推荐阅读顺序**：先看 §3.4 (simplified loss)，再回看 §3.1-§3.3 看是怎么来的。否则容易迷失在公式里。

---

## 三、核心公式（必背）

### 公式 1: Forward process
$$q(x_t | x_0) = \mathcal{N}\bigl(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I\bigr)$$

—— 任意 $t$ 一步加噪。

### 公式 2: Posterior
$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(\tilde\mu_t(x_t, x_0), \tilde\beta_t I)$$

—— 真实反向（贝叶斯反推）。

### 公式 3: $\epsilon$-prediction
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t, t)\right)$$

—— 网络参数化（关键洞察）。

### 公式 4: Simplified loss
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\|^2\right]$$

—— **整篇论文的精华**。

---

## 四、容易误读的地方

### 1. "Diffusion model 是 hierarchical VAE"

不少教程引用这句，但**容易误导**。
正确理解：**DDPM 的训练目标确实是 ELBO 的特殊形式**，但 forward "encoder" 是**固定**的（无参数）。所以"VAE"框架只是数学包装，工程上 DDPM 与 VAE 差异很大。

---

### 2. "$L_T$ 项被忽略不严谨"

§3.1 说 $L_T$ 设计上接近 0 所以可忽略。但严格说：
- $L_T = D_{\text{KL}}(q(x_T|x_0) \| p(x_T))$
- 只有当 $\bar\alpha_T \to 0$ 时 $L_T \to 0$
- 实际 $\bar\alpha_T \approx 0.0006$，$L_T$ 很小但不严格为 0

工程上忽略安全，但理论严谨论文（如 Improved DDPM）会保留。

---

### 3. "Simplified loss 抛弃了 ELBO 权重？"

**不是抛弃**，而是**经验上发现等权重更好**。论文 §3.4 写：
> "We use a simplified version of this loss... which we find to be of better sample quality."

后续 Improved DDPM 论文重新引入 hybrid loss 部分恢复了 ELBO 权重（用于学方差）。

---

### 4. "$x_0$ 输出层用 discrete log-likelihood？"

§3.3 解释了 $p_\theta(x_0|x_1)$ 用离散像素 likelihood 表达。**但实践中，simplified loss 直接对 $t=0$ 也用 MSE**，并不真正用 discrete likelihood。这是个"理论 vs 实践"的小 gap。

---

## 五、与其他工作的连接

- **NCSN (Song & Ermon 2019)**：DDPM 与 NCSN 数学上等价（参考 derive_03 §11）。两者独立提出，Score SDE (2021) 统一两者
- **Sohl-Dickstein 2015**：DDPM 的"祖师爷"，首次提出"非平衡热力学"风格的扩散思想。但当时实验效果不强
- **VAE (Kingma 2014)**：DDPM 的 ELBO 推导直接借用 VAE 的工具

---

## 六、表格速读

| 实验 | 结论 |
|------|------|
| Table 1 (sample quality) | DDPM beats GAN on FID for CIFAR-10 |
| Table 2 (NLL) | DDPM 的 NLL 与最好的 likelihood 模型可比 |
| Figure 5 (qualitative) | DDPM 生成多样性远高于 GAN |

---

## 七、思考题（用于自查）

1. 为什么"linear $\beta_t$" 从 $10^{-4}$ 到 $0.02$？这两个数怎么选的？（提示：保证 $\bar\alpha_T$ 接近 0 但 $\beta_1$ 不破坏小步加噪结构）

2. Algorithm 1 (training) 和 Algorithm 2 (sampling) 分别在哪些细节上做了简化？

3. §4.4 提到 DDPM 模型的 inductive bias 与 progressive lossy compression 有关。具体什么意思？（提示：U-Net 的层次结构对应不同 frequency band）

4. 为什么 sampling 时 $z = 0$ 当 $t = 1$？（提示：保证最后输出确定性）

---

## 八、引用方式（写论文用）

> "We follow the standard diffusion framework of \cite{ho2020denoising}, training on the simplified noise prediction loss..."

BibTeX:
```
@article{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={NeurIPS},
  year={2020}
}
```
