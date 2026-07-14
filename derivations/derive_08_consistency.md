# Derivation 08: Consistency Models 推导

> 对应 L13 内容。本手稿推导：
> - Consistency function 的存在性
> - Boundary condition 的参数化技巧
> - CD vs CT 的等价性（在 $N \to \infty$ 极限）

---

## §1 设置

probability flow ODE：
$$\frac{dx}{dt} = -\sigma(t) \nabla_x \log p_t(x), \quad t \in [\epsilon, T]$$

（VE-SDE 形式，$\epsilon > 0$ 是数值稳定的下界）

ODE 解轨迹 $\{x_t\}_{t \in [\epsilon, T]}$ 由 $x_T \sim \mathcal{N}(0, \sigma^2(T) I)$ 决定。

---

## §2 Consistency Function 定义

### 2.1 形式定义

$f: \mathbb{R}^d \times [\epsilon, T] \to \mathbb{R}^d$ 称为 **consistency function**，如果：

**Property 1 (Self-consistency)**：对同一轨迹上的两个点：
$$f(x_t, t) = f(x_{t'}, t'), \quad \forall (t, t')$$

**Property 2 (Boundary)**：
$$f(x_\epsilon, \epsilon) = x_\epsilon$$

---

### 2.2 存在性

**定理**：given probability flow ODE，consistency function 唯一存在。

**直觉证明**：定义 $f(x_t, t) = x_\epsilon$（沿轨迹追溯到起点）。这个定义满足两个 properties。

唯一性：两个 consistency function 都满足相同 property，在 $\epsilon$ 处都等于 $x_\epsilon$，由轨迹唯一性必相等。 ∎

---

## §3 Parameterization with Boundary Condition

### 3.1 朴素参数化

设 $f_\theta(x, t) = F_\theta(x, t)$（直接神经网络）。

**问题**：很难保证 $F_\theta(x_\epsilon, \epsilon) = x_\epsilon$。

---

### 3.2 EDM 式 preconditioning

Song et al. 2023 采用：
$$f_\theta(x, t) = c_{\text{skip}}(t) \cdot x + c_{\text{out}}(t) \cdot F_\theta(c_{\text{in}}(t) \cdot x, c_{\text{noise}}(t))$$

设计：
- $c_{\text{skip}}(\epsilon) = 1, c_{\text{out}}(\epsilon) = 0$ → boundary 自动满足
- $c_{\text{in}}, c_{\text{noise}}$ 用于输入归一化

具体（VE-SDE 设置）：
$$c_{\text{skip}}(t) = \frac{\sigma_{\text{data}}^2}{(t - \epsilon)^2 + \sigma_{\text{data}}^2}$$
$$c_{\text{out}}(t) = \frac{\sigma_{\text{data}} \cdot (t - \epsilon)}{\sqrt{t^2 + \sigma_{\text{data}}^2}}$$

---

### 3.3 验证 boundary

在 $t = \epsilon$ 处：
- $c_{\text{skip}}(\epsilon) = \sigma_{\text{data}}^2 / (0 + \sigma_{\text{data}}^2) = 1$ ✓
- $c_{\text{out}}(\epsilon) = \sigma_{\text{data}} \cdot 0 / \dots = 0$ ✓

所以 $f_\theta(x, \epsilon) = 1 \cdot x + 0 = x$ ✓

---

## §4 Consistency Distillation (CD)

### 4.1 训练目标

设 teacher 模型 $s_\phi$（已训好的 diffusion score）。

在离散化 $\epsilon = t_1 < t_2 < \dots < t_N = T$ 上：

$$\mathcal{L}_{\text{CD}}(\theta) = \mathbb{E}_{n, x_0, \epsilon}\left[d\bigl(f_\theta(x_{t_{n+1}}, t_{n+1}), f_{\theta_-}(\hat x_{t_n}, t_n)\bigr)\right]$$

其中：
- $x_{t_{n+1}}$ 由 forward process 加噪 $x_0$ 得到
- $\hat x_{t_n}$ 由 teacher 走一步 ODE: $\hat x_{t_n} = x_{t_{n+1}} - (t_{n+1} - t_n) \cdot s_\phi(x_{t_{n+1}}, t_{n+1}) \cdot t_{n+1}$
- $\theta_-$ 是 EMA copy of $\theta$
- $d$ 是距离函数（L2 或 LPIPS）

---

### 4.2 EMA Teacher 的作用

为什么用 $f_{\theta_-}$ 而不是 $f_\theta$ 本身？

**直觉**：避免训练不稳定（self-distillation 类似于 BYOL）。
- $f_\theta$ 更新太快，target 一直在动
- EMA $\theta_- = \mu \theta_- + (1-\mu) \theta$，target 平滑

实证：$\mu = 0.99$ 左右最优。

---

### 4.3 训练算法

```python
def cd_train_step(student, teacher, optimizer, ema_decay=0.99):
    x_0 = sample_data()
    n = randint(1, N - 1)
    t_n, t_np1 = noise_schedule[n], noise_schedule[n+1]
    
    eps = randn_like(x_0)
    x_tnp1 = x_0 + t_np1 * eps  # noise to t_{n+1}
    
    # Teacher走一步反向 ODE
    with torch.no_grad():
        score = teacher.score(x_tnp1, t_np1)
        x_tn = x_tnp1 - (t_np1 - t_n) * t_np1 * score
    
    # Student 在两个点的输出
    pred_np1 = student(x_tnp1, t_np1)
    with torch.no_grad():
        pred_n = student_ema(x_tn, t_n)
    
    loss = distance(pred_np1, pred_n)
    loss.backward()
    optimizer.step()
    
    # Update EMA
    for p, p_ema in zip(student.params, student_ema.params):
        p_ema.data = ema_decay * p_ema.data + (1 - ema_decay) * p.data
```

---

## §5 Consistency Training (CT)

CT 不需要 teacher。

### 5.1 关键 trick

Teacher 走一步 ODE 是为了**沿 ODE 轨迹**移动。CT 用一个 trick 模拟：

**同一个 noise**，加到不同的 $t$ 水平上：
$$x_{t_n} = x_0 + t_n \cdot \epsilon$$
$$x_{t_{n+1}} = x_0 + t_{n+1} \cdot \epsilon$$

（注意：同一个 $\epsilon$！）

这两个点**近似在同一 probability flow 轨迹上**——因为 forward SDE 在 VE-SDE 下与 noise injection 是一一对应的。

---

### 5.2 训练目标

$$\mathcal{L}_{\text{CT}}(\theta) = \mathbb{E}_{n, x_0, \epsilon}\left[d\bigl(f_\theta(x_0 + t_{n+1} \epsilon, t_{n+1}), f_{\theta_-}(x_0 + t_n \epsilon, t_n)\bigr)\right]$$

---

### 5.3 为什么 CT 等价于 CD（in limit）

**关键定理 (Song et al. 2023, Thm 4)**：在 $N \to \infty$ 的极限下，CT 和 CD 的训练 gradients 重合。

**直觉证明 sketch**：
- CD 用 teacher 走 ODE，得到 $\hat x_{t_n}$
- CT 直接用 $x_0 + t_n \epsilon$ 作为 $x_{t_n}$
- 当 $\Delta t = t_{n+1} - t_n \to 0$，teacher ODE 步退化为"$x_{t_{n+1}}$ 减去一点 score"
- 这恰好对应"同一 $\epsilon$ 下的 less noise"——即 $x_0 + t_n \epsilon$

数学细节：见 Song 2023 Appendix B.

---

## §6 距离函数选择

### 6.1 L2

最简单：$d(x, y) = \|x - y\|^2$。

但**视觉质量差**：L2 对低频敏感，高频不敏感。

### 6.2 LPIPS

$d(x, y) = \text{LPIPS}(x, y)$（用 pretrained VGG 抽特征算余弦距离）。

**视觉质量好**，但：
- 慢（每步多一次 VGG forward）
- 需要 decode 到 image space（latent CT 不直接适用）

### 6.3 Pseudo-Huber (Improved CT, Song 2024)

$$d(x, y) = \sqrt{\|x-y\|^2 + c^2} - c$$

- 小误差时近似 L2
- 大误差时近似 L1
- 不需要 VGG，便宜

实证：LPIPS 与 Pseudo-Huber 接近，但后者训练更稳定。

---

## §7 Multi-step CM

### 7.1 推理

虽然 CM 设计为 1-step，但**多步**采样质量更好：

```python
def cm_sample_multistep(model, n_steps=4):
    sigmas = sigma_schedule(n_steps)
    x = randn(...) * sigmas[0]  # noise at sigma_max
    for i in range(n_steps):
        # Consistency step
        x_0 = model(x, sigmas[i])  # 直接预测 x_0
        if i < n_steps - 1:
            # Re-add noise to next sigma level
            x = x_0 + sigmas[i+1] * randn_like(x)
        else:
            x = x_0
    return x
```

机制：每步 estimate $x_0$，然后**加更少**的 noise 重新进入扩散，再 estimate 一次。

### 7.2 与 DDIM 的差别

DDIM 每步**前进**（$t \to t-\Delta t$），CM 每步**直接预测 $x_0$**。

效果：CM 4 步 ≈ DDIM 50 步。

---

## §8 Loss Landscape 与稳定性

### 8.1 EMA teacher 的稳定作用

没有 EMA 的话：
- 一开始 $f_\theta$ 输出随机
- 让 $f_\theta(x_t)$ 与 $f_\theta(x_{t-\Delta t})$ 一致 = "和自己一致"
- 模式坍缩到 trivial 解（如 $f_\theta(x) = 0$）

EMA 让 target 滞后于 student，避免这个坍缩。

### 8.2 Spectral 分析

理论上可以分析 CT 的 stability：linear regime 下，learning rate 与 EMA decay 满足某不等式时收敛。详见 Song et al. 2024 Improved CT 附录。

---

## §9 自查题

1. 写出 consistency function 的两个 properties
2. 推导 EDM-style preconditioning 在 $t=\epsilon$ 时退化为 identity
3. 解释为什么 CT 在 $N \to \infty$ 下等价于 CD
4. 比较 L2、LPIPS、Pseudo-Huber 的优缺点

---

## §10 参考

- Song et al., *Consistency Models*, ICML 2023
- Song et al., *Improved Techniques for Training Consistency Models*, ICLR 2024
- Luo et al., *Latent Consistency Models*, 2023
