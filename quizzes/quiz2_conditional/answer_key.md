# Quiz 2 Answer Key

---

## 第一部分：概念题

### Q1
**CG**：
- 训 2 个网络：扩散模型 + 带噪声水平的分类器
- 工程上较麻烦（要训分类器）
- 数学公式：$\hat s = \nabla \log p + w \nabla \log p_\phi(y|x)$

**CFG**：
- 训 1 个网络（兼任条件与无条件）
- 工程上容易（只需在训练时做 conditional dropout）
- 公式：$\hat\epsilon = \epsilon_{\text{uncond}} + s(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$

**等价性**：数学上严格等价（CFG 用隐式分类器 $\nabla \log p^{\text{cond}} - \nabla \log p$ 替代 $\nabla \log p_\phi$）。

---

### Q2
**本质区别**：
- DDPM 是马尔可夫前向过程的反向，每步加随机性
- DDIM 是**非马尔可夫**前向的反向，可以确定性、可跳步

**解耦的重要性**：
- 训练一次的模型可服务多种推理需求（不同步数、确定/随机）
- 同一个网络可用于 DDIM、DPM-Solver、UniPC 等多种 sampler
- 工业实践中 inference 端可独立优化，不需要重训

---

### Q3
1. **边缘分布**：与原 SDE 在每个 $t$ 完全相同
2. **是否确定性**：是（没有随机项）
3. **是否可逆**：是（ODE 可正反两向积分）

这三个性质让 ODE 成为 DDIM、DPM-Solver、EDM 等加速 sampler 的母版。

---

### Q4
**来源**：
- SD 训完 VAE 后，对训练集所有 latent 计算 std
- $1 / \text{std} \approx 0.18215$（LAION 子集上的实测值）

**为什么需要**：
- VAE 的 latent 不严格是标准高斯（KL 权重很小）
- 实测 latent std $\approx 5.5$（远大于 1）
- 乘 0.18215 让 scaled latent 接近 std=1，匹配 DDPM 假设的 $\mathcal{N}(0, I)$ 先验

工程实践：自己训 LDM 必须在自己的数据上重新算这个值。

---

### Q5
SD 的 VAE 不是真正的"生成 VAE"，而是**轻微正则化的 autoencoder**：
- 目标不是从 prior 采样，而是提供"perceptually equivalent"的压缩
- 如果用标准 KL 权重（$\beta=1$），latent 会过度规整化，丢失图像细节
- 极小 KL ($10^{-6}$) 只起"鼓励 latent 大致接近高斯"的弱约束

> **更精确的说法**：SD 用的是"KL-VAE" 但其实更像 AutoEncoder + GAN + LPIPS。

---

### Q6
**错**。

虽然概念上"两次 UNet 前向"，但工程实现通过 batch concat（把 cond 与 uncond 拼成 batch=2 的输入）一次前向完成。GPU 并行使得速度**几乎不翻倍**（约 1.2-1.5 倍），显存翻倍。

但你说"显存代价大"是对的。

---

## 第二部分：推导题

### Q7
**反向 SDE**：
$$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] \, dt + g(t) \, d\bar W$$

**推导思路**：
1. 正向 FPE：$\partial_t p_t = -\nabla \cdot (f p_t) + \frac{1}{2} g^2 \Delta p_t$
2. 反向 FPE 形式应当满足 $-\partial_t p_t = \dots$（时间反向）
3. 用恒等式 $\Delta p_t = \nabla \cdot (p_t \nabla \log p_t)$，对比两个 FPE，得反向 drift = $f - g^2 \nabla \log p_t$

（参考 derive_04 §3）

---

### Q8
**约束 1（均值）**：边际化恒等式 $q(x_{t-1}|x_0) = \int q(x_{t-1}|x_t, x_0) q(x_t|x_0)$ 在均值上给出：
$$a_t + b_t \cdot \sqrt{\bar\alpha_t} = \sqrt{\bar\alpha_{t-1}}$$

即 $a_t = \sqrt{\bar\alpha_{t-1}} - b_t \sqrt{\bar\alpha_t}$。

**约束 2（方差）**：在方差上：
$$\sigma_t^2 + b_t^2 (1-\bar\alpha_t) = 1 - \bar\alpha_{t-1}$$

即 $b_t = \sqrt{\frac{1 - \bar\alpha_{t-1} - \sigma_t^2}{1 - \bar\alpha_t}}$。

代入 $a_t$ 得：
$$a_t = \sqrt{\bar\alpha_{t-1}} - \sqrt{\bar\alpha_t \cdot \frac{1-\bar\alpha_{t-1}-\sigma_t^2}{1-\bar\alpha_t}}$$

（参考 derive_05 §2）

---

### Q9
**Step 1**：从贝叶斯有：
$$\nabla_x \log p(x|y) = \nabla_x \log p(y|x) + \nabla_x \log p(x)$$

反推：
$$\nabla_x \log p(y|x) = \nabla_x \log p(x|y) - \nabla_x \log p(x)$$

**Step 2**：CG 公式（加权 score）：
$$\hat s = \nabla_x \log p(x) + w \nabla_x \log p(y|x)$$

**Step 3**：把 Step 1 代入：
$$\hat s = \nabla_x \log p(x) + w [\nabla_x \log p(x|y) - \nabla_x \log p(x)]$$
$$= (1-w) \nabla_x \log p(x) + w \nabla_x \log p(x|y)$$

**Step 4**：用 noise-score 关系 $s = -\epsilon / \sqrt{1-\bar\alpha_t}$：
$$\hat\epsilon = (1-w) \epsilon_{\text{uncond}} + w \epsilon_{\text{cond}}$$
$$= \epsilon_{\text{uncond}} + w (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$$

设 $s = w$，即得 CFG 公式。

---

### Q10
ODE 对应的 transport equation：
$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (h \cdot p_t), \quad h = f - \frac{1}{2}g^2 \nabla \log p_t$$

原 SDE 的 FPE：
$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2} g^2 \Delta p_t$$

用恒等式 $\Delta p_t = \nabla \cdot (p_t \nabla \log p_t)$：
$$\frac{\partial p_t}{\partial t} = -\nabla \cdot \left[(f - \frac{1}{2} g^2 \nabla \log p_t) p_t\right] = -\nabla \cdot (h \cdot p_t)$$

—— 两个 transport equation 同形，初值 $p_0$ 相同，所以 $p_t$ 对所有 $t$ 相同。 ∎

（参考 derive_04 §4）

---

## 第三部分：实验设计题

### Q11
(a) **CFG scale 过高**。典型 SD 默认 7.5 是安全选择；如果用户写了 15+ 会出现这种"过饱和、过典型"的特征。

(b) 两个方案：
1. **降低 CFG**：7.5 → 4.0-5.0。代价是 prompt fidelity 略降，但视觉自然
2. **dynamic thresholding**（Imagen 思路）：保持 CFG=7.5，但在采样中把 $\hat x_0$ clip 到 99.5 percentile。这能在高 CFG 下保留质量

(c) 用 negative prompt：
- `"low quality, blurry, oversaturated, deformed, overexposed"`
- 把 $\emptyset$ 换成上面的 negative，让 score 远离这些特征
- 副作用：可能多样性下降；不如直接降 CFG 直接

---

### Q12
(a) 4 个优化方向：
1. **fp16 / bf16**（最大收益，30-50%）
2. **xformers / Flash Attention**（25-30%）
3. **DPM-Solver++** sampler（步数从 50 → 20，~60% 加速）
4. **VAE/Attention slicing**（显存优化，间接加速）
5. **torch.compile**（10-20%，需要 PyTorch 2.0+）
6. **TensorRT 编译**（30-50%，但工程复杂）

(b) 代码动作（diffusers）：
```python
pipe = pipe.to(device, dtype=torch.float16)            # fp16
pipe.enable_xformers_memory_efficient_attention()       # xformers
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# inference 时 num_inference_steps=20
```

(c) 30s → 5s 的权衡：
- **质量**：DPM-Solver 20 步 vs DDIM 50 步，肉眼差不多但 FID 略升 1-2
- **精度**：fp16 在某些场景出现 NaN（罕见，可加 fp32 fallback）
- **多样性**：sampler 偏向 ODE，多样性略降
- **代码复杂**：xformers/TensorRT 引入依赖，部署链路变长

---

## 评分参考

- 推导题：完整步骤 + 正确结论 = 满分；缺一步扣 2-3 分
- 概念题：抓住关键 + 不出错 = 满分
- 实验题：方向对 + 具体动作 = 满分；只说"试试不同 CFG"无具体方案扣分
