# L13: Consistency Models 与一步生成（W12）

> **目标**：理解"一步生成"的可能性 —— 实时图像生成与机器人控制的关键
> **前置**：L06 (Score SDE), L12 (Flow Matching)
> **核心问题**：DDPM 50 步、FM 10 步、能否再压缩到 **1 步**？

---

## §1 课程定位

如果 DDIM/DPM-Solver 是"少 NFE 革命"，Consistency Model 是"**单 NFE 革命**"。

为什么重要：
- **实时图像生成**（30 FPS 视频流编辑）
- **机器人控制**（20 Hz 实时 action 生成）
- **Edge 部署**（手机端 SD）

代表工作：
- Consistency Models (Song et al., 2023)
- LCM / Latent Consistency Models (Luo et al., 2023)
- SDXL Turbo, SD 3 Turbo
- Consistency Distillation 框架

---

## §2 一步生成的挑战

### 2.1 GAN 是一步的，为什么我们要换 diffusion？

GAN 一步生成但训练不稳定、多样性差。

Diffusion 训练稳定但要多步采样。

**Consistency Models 想要"两个世界的好处"**：训练像 diffusion 一样稳定，推理像 GAN 一样一步。

---

### 2.2 简单回顾：probability flow ODE

```
x_T (noise) ────ODE────→ x_0 (data)
       ↑                    ↑
       从这里出发        到这里
```

ODE 是确定性的——给定 $x_T$，存在唯一 $x_0$。

**一步生成的等价问题**：能否学一个函数 $f_\theta(x_T) \approx x_0$？

---

## §3 Consistency Function 定义

### 3.1 关键定义

设 $\{x_t\}$ 是 probability flow ODE 在 $[\epsilon, T]$ 上的解轨迹。**Consistency function** $f$ 满足：

$$f(x_t, t) = f(x_{t'}, t'), \quad \forall t, t' \in [\epsilon, T]$$

—— **同一条 ODE 轨迹上的所有点都映射到同一目标**。

特别地：$f(x_t, t) = x_\epsilon$（轨迹起点附近）。

---

### 3.2 边界条件

$$f(x_\epsilon, \epsilon) = x_\epsilon$$

（在 $t = \epsilon$ 处是 identity）

这是 self-consistency 的基线。

---

### 3.3 参数化技巧

直接学 $f_\theta(x, t) \approx x_\epsilon$ 会忽略 $t = \epsilon$ 的边界条件。Song et al. 提出：

$$f_\theta(x, t) = c_{\text{skip}}(t) \cdot x + c_{\text{out}}(t) \cdot F_\theta(x, t)$$

其中 $c_{\text{skip}}(\epsilon) = 1, c_{\text{out}}(\epsilon) = 0$，保证 $f_\theta(x, \epsilon) = x$。

EDM 风格的 preconditioning（参考 L11）：
- $c_{\text{skip}}(t) = \sigma_{\text{data}}^2 / (\sigma_{\text{data}}^2 + t^2)$
- $c_{\text{out}}(t) = \sigma_{\text{data}} \cdot t / \sqrt{\sigma_{\text{data}}^2 + t^2}$

---

## §4 训练方式 1：Consistency Distillation (CD)

### 4.1 思路

已有训好的 diffusion model $s_\phi$（teacher）。让 $f_\theta$（student）"复制"它的 ODE 行为，但能一步到位。

---

### 4.2 算法

```
1. 采样 x_0 ~ data, n ~ U[1, N-1]
2. 在 t_{n+1} 处加噪: x_{t_{n+1}} = x_0 + t_{n+1} * eps
3. 用 teacher 走一步 ODE: x_{t_n} = ODE_step(x_{t_{n+1}}, t_{n+1} → t_n, s_phi)
4. Loss = d(f_theta(x_{t_{n+1}}, t_{n+1}), f_theta_EMA(x_{t_n}, t_n))
```

**关键**：
- $f_\theta$ 与 $f_{\theta_{\text{EMA}}}$ 是同一个网络（EMA 副本）
- 让 $f_\theta(x_{t_{n+1}}, t_{n+1})$ 与 $f_\theta(x_{t_n}, t_n)$ 一致 ——**self-consistency!**

距离函数 $d$：LPIPS 或 L2。

---

### 4.3 推理

```python
@torch.no_grad()
def consistency_sample(model, n_steps=1, sigma_max=80):
    x = torch.randn(...) * sigma_max  # noise scaled
    if n_steps == 1:
        return model(x, sigma_max)  # 一步！
    else:
        # Multi-step (improves quality)
        for sigma in sigma_schedule(n_steps):
            x_0 = model(x, sigma)
            x = x_0 + sigma_next * torch.randn_like(x)
        return x_0
```

---

### 4.4 实验结果

CIFAR-10（Song et al. 2023）：
| NFE | FID |
|-----|-----|
| Teacher EDM (35) | 1.79 |
| **CD-distilled (1)** | **2.83** |
| CD-distilled (2) | 2.20 |

—— 单步 FID 2.83，已经接近 multi-step diffusion。

---

## §5 训练方式 2：Consistency Training (CT)

CD 需要预训好的 teacher，CT **不需要**——从头训。

### 5.1 算法

把"teacher 走一步 ODE"换成"加入更多噪声"：
```
1. 采样 x_0 ~ data, n ~ U[1, N-1]
2. 同一个 x_0，加不同噪声: 
   x_{t_n} = x_0 + t_n * eps
   x_{t_{n+1}} = x_0 + t_{n+1} * eps  # 同一个 eps
3. Loss = d(f_theta(x_{t_{n+1}}, t_{n+1}), f_theta_EMA(x_{t_n}, t_n))
```

直觉：两个噪声水平的样本来自同一 $x_0$，consistency function 应该给同样结果。

---

### 5.2 CT vs CD

| 维度 | CD | CT |
|------|----|----|
| 需要 teacher | 是 | 否 |
| 训练成本 | 低（finetune） | 高（从头训） |
| 质量 | 高 | 略低 |
| 适用场景 | 已有 diffusion 时 | 没有时 |

工程上 CD 是常规选择（因为 SD 等 teacher 普遍可得）。

---

## §6 Latent Consistency Model (LCM)

Luo et al. 2023 把 CD 应用到 **latent diffusion**（SD）：

### 6.1 关键设计

1. 用 SD 1.5 作为 teacher
2. 在 latent space 做 CD distillation
3. 加入 LoRA：只训 LoRA 参数（不动 SD UNet）
4. 加入 CFG：teacher 的 ODE step 用 CFG=8 outputs

### 6.2 LCM-LoRA

特别值得讲：**LCM-LoRA 是一个 LoRA**，可以直接套到任何 SD 1.5 fine-tune 上：
- 装上 LCM-LoRA → 4 步生成
- 卸下 LCM-LoRA → 50 步生成（高质量）

实际部署中 LCM-LoRA 让 SD 实时化（手机 6 步生成 512×512）。

---

## §7 后续工作：Multistep Consistency

Song et al. 2024 后续（**Improved Techniques for CT**）提出：

### 7.1 Schedule 改进
- $N$ 步数从 18 → 1280（更精细）
- 自定义 $\sigma$ schedule（不再用 EDM 默认）

### 7.2 Pseudo-Huber Loss
$$d(x, y) = \sqrt{\|x-y\|^2 + c^2} - c$$
比 L2 更稳定，比 LPIPS 更便宜。

### 7.3 EMA Teacher Trick
Teacher network 用学生的 EMA 副本（self-distillation in continuous form）。

实验：CT @ 2 NFE 达到 CD @ 1 NFE 同样质量。

---

## §8 与 VLA / Embodied 的连接

### 8.1 实时机器人控制

机器人要求 20-50 Hz 控制频率：
- 标准 diffusion policy: 50 steps × 5 ms/step = 250 ms → **不可用**
- LCM distilled policy: 4 steps × 5 ms = 20 ms → **达标**

具体应用：
- **DiPo (Diffusion Policy)**：原版 50 步
- **LCM-DiPo / Streaming Diffusion Policy**：CD 蒸馏到 1-4 步

### 8.2 流式生成（streaming）

世界模型 / 视频生成中，consistency model 让"逐帧产出"成为可能：
- 不需要等整个 video 生成完
- 实时反馈到 controller

**Streaming Diffusion (Kodaira et al., 2023)** 在视频生成上用 LCM 实现 30 FPS。

---

## §9 课后任务

### 必做
1. 跑通 nb10（CD distillation on MNIST/CIFAR）
2. 比较 LCM 4 步与原 SD 50 步的视觉质量
3. 实现一个 multi-step CT loop（理解 EMA teacher）

### 进阶
4. 把 LCM-LoRA 卸载到自己训的 SD fine-tune，验证 plug-and-play 性质
5. 思考：CT 与 GAN 的等价性（hint: minimax）
6. 读 Improved CT (Song 2024) 论文，理解 pseudo-Huber loss 的选择

---

## §10 FAQ

**Q1：Consistency Model 是 GAN 吗？**

A：**不是**。GAN 用 discriminator，CM 没有；GAN 的 loss 是对抗的，CM 是 self-consistency 的 MSE。但它们都能"一步生成"，所以**直觉上接近**。

事实上，CT 的某些数学性质可以解释为"隐式 GAN"，但训练动力学完全不同。

---

**Q2：CD 的 student 能超过 teacher 吗？**

A：**不能**。CD 是 distillation，student 在 FID 上严格 ≥ teacher（更差或持平）。

但 CD 的**速度优势**远大于这点 FID 损失。工程上是赚的。

---

**Q3：为什么 LCM-LoRA 这么牛？**

A：因为 SD 1.5 的 fine-tune 生态太大（几万个）。每个 fine-tune 单独蒸馏要训成千上万次。LCM-LoRA 设计成 **plug-and-play**，一次蒸馏，对所有 SD 1.5 fine-tune 有效。

---

**Q4：Consistency 与 Flow Matching 谁是未来？**

A：**互补**。
- FM 是更好的训练框架（替代 DDPM）
- Consistency 是更好的蒸馏框架（FM 训完后蒸到 1 步）
- 工业流水线：FM 训 base → CD 蒸馏到 1-4 步

---

## §11 参考资源

- Song et al., *Consistency Models*, ICML 2023（**原论文**）
- Song et al., *Improved Techniques for Training Consistency Models*, ICLR 2024
- Luo et al., *Latent Consistency Models*, 2023
- Sauer et al., *Adversarial Diffusion Distillation* (SDXL Turbo), 2023
- Black et al., *Pi-0* 中的 action distillation 章节

---

> 下一讲（L14-L15）我们从图像扩展到**视频**——Sora 的内核。
