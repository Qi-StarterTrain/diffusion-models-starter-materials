# 论文导读 15: Consistency Models (Song et al., ICML 2023) + LCM

**标题**：Consistency Models
**作者**：Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever (OpenAI)
**核心地位**：⭐⭐⭐⭐⭐ 一步生成的里程碑，工业部署核心技术

---

## 一、为什么必读

- 把扩散从 50 步 → 1 步
- 实时图像/视频生成的关键
- 后续 LCM、SDXL Turbo、SD 3 Turbo 都基于此

---

## 二、阅读路线图

| 章节 | 优先级 |
|------|--------|
| §1 Intro | ✅ |
| **§2 Consistency Models** | 🔴 核心 |
| §2.1 Definition | 🔴 核心 |
| §2.2 Parameterization | 🔴 核心 |
| §2.3 Sampling | 🔴 核心 |
| **§3 Training** | 🔴 核心 |
| §3.1 CD | 🔴 核心 |
| §3.2 CT | 🔴 核心 |
| §4 Experiments | ✅ |

---

## 三、核心公式

### Consistency Function
$f: \mathbb{R}^d \times [\epsilon, T] \to \mathbb{R}^d$，满足：
- **Self-consistency**：$f(x_t, t) = f(x_{t'}, t')$ 对同一 ODE 轨迹的点
- **Boundary**：$f(x_\epsilon, \epsilon) = x_\epsilon$

### EDM-style Preconditioning
$$f_\theta(x, t) = c_{\text{skip}}(t) x + c_{\text{out}}(t) F_\theta(c_{\text{in}}(t) x, c_{\text{noise}}(t))$$

设计 $c$ 函数让 $f_\theta(x, \epsilon) = x$ 自动满足。

### CD Loss
$$\mathcal{L}_{\text{CD}} = \mathbb{E}_n\left[d(f_\theta(x_{t_{n+1}}, t_{n+1}), f_{\theta^-}(\hat x_{t_n}, t_n))\right]$$

其中 $\hat x_{t_n}$ 由 teacher 走一步 ODE 得到。

### CT Loss
$$\mathcal{L}_{\text{CT}} = \mathbb{E}_n\left[d(f_\theta(x_0 + t_{n+1} \epsilon, t_{n+1}), f_{\theta^-}(x_0 + t_n \epsilon, t_n))\right]$$

没有 teacher，用同一 $\epsilon$ 加到不同噪声水平。

---

## 四、CD vs CT 等价性

定理：$N \to \infty$ 极限下 CD 与 CT gradient 重合。

实践：
- CD 训得更稳（用 teacher 作监督）
- CT 计算成本低（不需 teacher）
- 工业上 CD 居多（teacher 普遍可得）

---

## 五、实验结果

CIFAR-10：

| Model | NFE | FID |
|-------|-----|-----|
| EDM (teacher) | 35 | 1.79 |
| **CD (1 NFE)** | **1** | **2.83** |
| CD (2 NFE) | 2 | 2.20 |
| CT (1 NFE) | 1 | 8.70 |

CT 单独比 CD 差，但训练简洁。

---

## 六、LCM (Luo et al., 2023)

**Latent Consistency Models**：CD 应用到 SD UNet（latent space）。

关键工程改动：
1. 在 latent space distill SD 1.5
2. 用 LoRA 训（不改 base SD）→ **LCM-LoRA**
3. 4-8 步 NFE 替代原 50 步

**LCM-LoRA** 最有影响：插到任意 SD 1.5 fine-tune 上 → 4 步生成。

---

## 七、改进版：Improved CT (Song 2024)

后续工作《Improved Techniques for Training Consistency Models》：
- Pseudo-Huber loss（替代 LPIPS）
- 更精细的 sigma schedule（N: 18 → 1280）
- EMA teacher 的优化

效果：CT 2 NFE 达到 CD 1 NFE 同等质量。

---

## 八、常被误读

### 1. "CM 是 GAN？"
**不是**。CM 无 discriminator，loss 是 self-consistency。但训完模型**一步生成**这一点像 GAN。

### 2. "CM 一定要从 teacher 蒸馏？"
**否**。CT 不需要。但实践中 CD 质量更好。

### 3. "LCM-LoRA 是 LoRA？"
**Yes**——它就是按 LoRA 格式存储的 distillation 结果。可以与其他 LoRA 一起 stack。

---

## 九、思考题

1. 为什么 CT 的 1-NFE FID 8.7 显著差于 CD 的 2.83？根本原因？
2. LCM-LoRA 为什么能 plug-and-play 到任意 SD 1.5 fine-tune？
3. 实现 multi-step CM sampling，与 1-step 对比
4. 思考：CM 能用于 Flow Matching 模型吗？怎么改？

---

## 十、引用

```
@inproceedings{song2023consistency,
  title={Consistency Models},
  author={Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  booktitle={ICML},
  year={2023}
}
@article{luo2023latent,
  title={Latent Consistency Models},
  author={Luo, Simian and Tan, Yiqin and Huang, Longbo and Li, Jian and Zhao, Hang},
  year={2023}
}
```
