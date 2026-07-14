# Quiz 3 答案钥匙（W10-W14）

> **重要**：这只是参考答案，不是唯一答案。学生若答出等价或更深刻的内容应给满分。

---

## 一、概念题（30 分）

### 1. AdaLN-Zero "zero init" 含义

把 AdaLN 的 condition 投影层（输出 $\gamma, \beta, \alpha$）权重 init 为 0。

**必要性**：
- 让 $\gamma = \beta = \alpha = 0$ → modulate $(1+\gamma) \text{LN}(x) + \beta = \text{LN}(x)$，残差缩放 $\alpha \cdot \text{Attn} = 0$
- 整个 block 退化为 identity (just LN，加上 zero 残差)
- 训练初始大模型不爆炸，逐层"激活"

类比 ResNet 的 identity skip。

---

### 2. FM vs DDPM 本质区别

| | DDPM | Flow Matching |
|---|------|----------------|
| 路径 | SDE 加噪 | ODE 路径 (linear) |
| Schedule | $\bar\alpha_t$ 非线性 | $t$ 线性 |
| 采样 | 数千步 SDE | 几十步 ODE |
| 目标 | 噪声 $\epsilon$ | 速度 $v = x_1 - x_0$ |

**工程简洁性**：FM 更简洁——loss 就是 $\|v_\theta - (x_1 - x_0)\|^2$，没有 $\beta$ schedule、没有 $\bar\alpha$、没有 noise schedule 复杂数学。

---

### 3. Consistency Model vs GAN

| | GAN | CM |
|---|----|----|
| 损失 | 对抗（min-max） | self-consistency MSE |
| 训练稳定性 | 难（mode collapse 等） | 稳定 |
| 多样性 | 受 G 限制 | 由 prior 控制 |
| 是否需 teacher | 否 | CD 需要，CT 不需要 |

CM 是"扩散稳定性 + GAN 单步"的折中。

---

### 4. Spatial + Temporal factorized attention

**计算优势**：
- Full 3D: $O(N^2)$ where $N = THW$
- Factorized: $O(T \cdot (HW)^2 + HW \cdot T^2) \approx O(N^2 / \min(T, HW))$
- 在 video 上节省 32-100×

**表达能力代价**：
- 无法直接 attend "frame 5 位置 (3,7) → frame 8 位置 (12, 4)" (跨时空 + 跨空间)
- 需要至少 2 层 (spatial + temporal) 才能近似 full attention 的依赖
- 经验上 4-8 层后能 cover 大部分 patterns

---

### 5. Sora spacetime patches

把 $(T, H, W)$ 视频 latent 划分成 $(p_t, p_h, p_w)$ 大小的 3D 块，每块 flatten 为一个 token。

**支持任意分辨率/时长**：
- DiT 架构对 token 数量不敏感（self-attention 处理任意长度）
- Position embedding 用 sinusoidal / RoPE 3D，外推到没训过的 (T, H, W)
- Patchify 可 batch 不同 size 的视频（同一模型）

---

### 6. ControlNet zero conv & LoRA B=0 init

**共通点**：训练开始时新模块**不影响**原模型 forward。

**保护的东西**：
- ControlNet：保护 SD UNet 的 frozen 权重的语义
- LoRA：保护原 LLM/SD 的预训练能力

设计哲学：**"先做 identity，再学差异"**——同 ResNet 的 skip connection，同 AdaLN-Zero。

---

## 二、推导题（40 分）

### 7. Linear-path FM 推导（10 分）

**(a) 推导 $u_t(x_t | x_1)$**：

设 $x_t = (1-t) x_0 + t x_1$。对 $t$ 求全微分：
$$\frac{d x_t}{dt} = -x_0 + x_1 = x_1 - x_0$$

由 $u_t(x_t|x_1) \equiv \frac{d x_t}{dt}$（条件 ODE 的速度场），所以 $u_t(x_t|x_1) = x_1 - x_0$。

---

**(b) CFM loss**：
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U[0,1], x_1 \sim p_{\text{data}}, x_0 \sim \mathcal{N}(0, I)}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

其中 $x_t = (1-t) x_0 + t x_1$。

---

**(c) 等价性直觉**：

CFM loss 与 FM loss 数值上差一个常数（与 $\theta$ 无关）：
$$\mathcal{L}_{\text{CFM}} - \mathcal{L}_{\text{FM}} = \mathbb{E}[\|u_t(x|x_1)\|^2 - \|u_t(x)\|^2]$$

—— 这一项不依赖 $\theta$，所以对 $\theta$ 偏导为 0。两个 loss 的 gradient 相同。

详细证明用了 marginalization lemma：$u_t(x) = \mathbb{E}_{x_1|x}[u_t(x|x_1)]$。

---

### 8. CM EDM 参数化（10 分）

**(a) $c$ 函数作用**：
- $c_{\text{in}}$：归一化输入（让 $c_{\text{in}} \cdot x$ 在不同 $t$ 下分布一致）
- $c_{\text{noise}}$：把 $t$ 转成 sensitive 范围（如 $\log$）
- $c_{\text{skip}}, c_{\text{out}}$：boundary parametrization

**在 $t = \epsilon$ 处必须**：$c_{\text{skip}}(\epsilon) = 1, c_{\text{out}}(\epsilon) = 0$ → $f_\theta(x, \epsilon) = x$ 自动满足。

**(b) CD loss**：

$$\mathcal{L}_{\text{CD}} = \mathbb{E}_{n, x_0, \epsilon}\bigl[d(f_\theta(x_{t_{n+1}}, t_{n+1}), f_{\theta^-}(\hat x_{t_n}, t_n))\bigr]$$

其中 $\hat x_{t_n}$ 是 teacher 走一步 ODE 得到。

**EMA teacher 作用**：
- 学生与"自己的 EMA copy"对齐，避免 trivial 解（$f = 0$）
- 类似 BYOL / DINO 的 momentum target

**(c) CT trick**：

CT 用"同一个 noise $\epsilon$，加到不同的 $t$ 水平上"：
$$x_{t_n} = x_0 + t_n \epsilon, \quad x_{t_{n+1}} = x_0 + t_{n+1} \epsilon$$

(同一 $\epsilon$！) 

这两点近似在同一 ODE 轨迹上（VE-SDE 下精确）。在 $N \to \infty$ 极限下 CT gradient = CD gradient。

---

### 9. DiT FLOPs（10 分）

**(a) Token 数与 attention FLOPs**：
- $N = (32/2)^2 = 256$ tokens
- Self-attention FLOPs（一个 block）：$2 N^2 d$（Q@K + softmax(QK)@V，乘 2 算 attn 的两次矩阵乘）
- $= 2 \cdot 256^2 \cdot 1152 = 1.51 \times 10^8 \approx 0.15$ GFLOPs

**(b) FFN FLOPs（hidden 4d）**：
- 每 token：$2 \cdot d \cdot 4d + 2 \cdot 4d \cdot d = 16 d^2 \approx 16 \cdot 1152^2 \approx 2.12 \times 10^7$
- 全 N tokens：$N \cdot 16 d^2 = 256 \cdot 2.12 \times 10^7 = 5.43 \times 10^9 \approx 5.4$ GFLOPs

—— FFN 主导（35× attention）。

**(c) Total training FLOPs**：
- 每 forward block: $0.15 + 5.4 \approx 5.55$ GFLOPs
- 28 layers: $28 \times 5.55 = 155$ GFLOPs per image
- Backward 约 2× forward → $465$ GFLOPs per image-step
- Batch 256 × 1M steps: $256 \cdot 10^6 \cdot 465 \times 10^9 = 1.19 \times 10^{20}$ FLOPs

对比 SD UNet 256×256 @ 0.5 TFLOPs/image-forward：
- DiT-XL/2 forward: $155 / 0.5 \times 10^3 \approx 0.31$ → DiT 单步 forward **比 SD UNet 便宜约 3×**（同 latent resolution）

但 SD 256×256 在 64×64 latent (8x downsample) 上做 attention；DiT 在 32×32 latent 上 patch 后只有 16×16=256 tokens。具体比例依配置。

---

### 10. Video attention factorization（10 分）

设 $N = 32 \cdot 32 \cdot 32 = 32768$。

**(a) Full 3D**：
- Attention matrix: $N \times N = 32768^2 \approx 1.07 \times 10^9$
- fp16: 2 bytes/entry → $\approx 2.14$ GB **per attention layer per head!**
- 多头 + 多层 → 数百 GB → 不可行

**(b) Factorized**：
- Spatial: $T$ 个独立 attention, 每个 $(HW)^2 = (32 \cdot 32)^2 = 1.05 \times 10^6$ entries
- Total spatial: $32 \cdot 1.05 \times 10^6 = 3.35 \times 10^7$
- Temporal: $HW$ 个独立, 每个 $T^2 = 32^2 = 1024$ entries
- Total temporal: $1024 \cdot 1024 = 1.05 \times 10^6$
- Grand total: $3.46 \times 10^7$ entries × 2 bytes = **69 MB** per layer per head

**节省**：$2.14 \times 10^9 / 6.9 \times 10^7 \approx 31\times$ ✓

**(c) Window attention** ($w_t \times w_h \times w_w = 4 \times 8 \times 8$)：
- 每 window: $w_t \cdot w_h \cdot w_w = 256$ tokens → $256^2 = 65536$ entries
- 窗口数: $N / 256 = 32768 / 256 = 128$
- Total: $128 \cdot 65536 = 8.4 \times 10^6$ entries × 2 = **16.8 MB**

**节省**：$2.14 \times 10^9 / 8.4 \times 10^6 \approx 255\times$ ✓

---

## 三、实验设计题（30 分）

### 11. FM vs DDPM 公平对比（15 分）

**(a) 实验设置**：

- 数据集：CIFAR-10（small）+ ImageNet 64×64（medium）+ LAION subset 1M（large）
- 模型：DiT-S 与 DiT-B 各一份，one trained as DDPM、another as FM；保持 architecture 完全相同
- 训练规模：每个 setting 训 200K 步，相同 batch=256、相同 EMA decay
- 重复：每个 setting 3 个不同 seed
- 控制：相同 augmentation、相同 LR schedule、相同 grad clip

**(b) 评估指标**：
1. **FID**：标准生成质量。但只看 FID 有限——
2. **CLIP-Score**：text-image alignment（若 conditional）
3. **NFE-FID Pareto**：FID 在 NFE = 5, 10, 20, 50, 100, 250 下的曲线——FM 应在 low NFE 区间胜出
4. **训练 loss 收敛速度**：相同步数下哪个 loss 更低
5. **Wallclock time per FID-1 improvement**：实际工程效率

**(c) 预期结果与反驳**：

期望观察：
- FM 在 small data + low NFE 下优势明显
- 大 model + 大 data 下 FM 与 DDPM 的 FID 接近（< 5% 差距）
- 在 NFE > 100 时两者 FID 几乎一致

反驳"FM 一定优于 DDPM"：
- 如果 model 容量足够，DDPM 也能学到 ≈ direct 的 trajectories（through schedule tuning）
- EDM 论文（DDPM 变种）的 FID 与 FM 相当
- 工程优势 ≠ 理论优势
- "FM 优势"主要在 **少步采样**与**实现简洁**，不在最终 FID 

---

### 12. VLA Action Diffusion 设计（15 分）

**(a) Action 维度与 chunk size**：

设 robot 是 7-DoF arm（6 DoF pose + 1 gripper）。
- Action dim: 7
- 频率：50 Hz 控制
- Chunk size：50 步 (1 秒未来)

为什么 50：
- 长 horizon 利于 planning（多步 lookahead）
- 50 步 × 7 dim = 350 维 → 模型可处理
- 与 Pi-0 一致

**(b) DDPM vs FM vs CM**：

| | DDPM | FM | CM |
|---|------|----|----|
| 控制频率 | 1-5 Hz | 50-100 Hz | 100+ Hz |
| 训练成本 | 中 | 中 | 高（需 teacher） |
| 质量 | 中 | 高 | 中 |

**选择 FM**：
- 控制频率达标（50 Hz 必须）
- 实现简洁
- 训练成本可控
- Pi-0 已验证可行

不选 DDPM：太慢
不选 CM：训练成本高，且需要先训好的 teacher（鸡生蛋问题）

**(c) 评估指标 + ablation**：

3 个评估指标：
1. **成功率**：经典 success/total
2. **轨迹平滑度**：jerk = ||a_{t+1} - a_t||² 平均，越小越平滑
3. **多模态保留**：fix obs，反复采样 100 次，看 action variance（multimodal 的健康 sign）

Ablation：
- 比较 chunk size 16 / 32 / 50 / 100，看长 horizon 是否真的帮助
- 比较 image-only vs image+proprio，看 proprio 增益
- 比较 FM 4 步 vs 10 步 vs 30 步采样，看 NFE-success Pareto

---

## 评分准则汇总

- 每题答出**关键技术点**得部分分
- 推导题量纲对、结论对即可（即使中间步骤简化）
- 实验设计题鼓励**自我批判**与**对照实验**
- 创新答案（论文水准的）可超出满分上限作教学评语
