# Quiz 1 Answer Key

> 评分指南：完整答出关键步骤即可满分。次要细节缺失扣 1-2 分。

---

## 第一部分：概念题

### Q1
- $q(x_t|x_0)$：从干净数据**直接**加噪到 $t$ 时刻，闭合高斯
- $q(x_t|x_{t-1})$：单步加噪（forward 过程的定义）
- $q(x_{t-1}|x_t, x_0)$：**真实反向后验**（贝叶斯反推，闭合高斯）

DDPM 训练用 $q(x_t|x_0)$。原因：
1. 闭合形式让我们能直接对任意 $t$ 采样 $x_t$，无需逐步加噪
2. simplified loss 只依赖 $q(x_t|x_0)$ 不依赖完整链——这是后续 DDIM 能用同一模型的根基

---

### Q2
$\mathcal{L}_{\text{simple}}$ = ELBO 推出的 loss 但**去掉了时间相关权重 $w_t$**。

DDPM 用前者原因：
- 实验上 FID 更好
- 等权重让大 $t$ 的"困难时间步"得到充分训练，避免被小 $t$ 主导
- 简洁、稳定

---

### Q3
**错**。DDPM 标准 ancestral sampling 在每步中加入噪声 $z \sim \mathcal{N}(0, I)$，所以输出**随机**。即使固定初始 $x_T$，每次结果也不同。

确定性采样要用 DDIM ($\sigma = 0$) 或 probability flow ODE。

---

### Q4
1. $\tilde\mu_t$ 自然形式中显含 $\epsilon$，预测 $\epsilon$ 给出**简洁的 loss 形式**（MSE）
2. $\epsilon$ 的数值范围（标准正态 $[-3, 3]$）天然适配神经网络回归
3. 预测 $\epsilon$ 等价于预测 score（用于 SDE/ODE 视角）
4. 实验上 FID 比预测 $x_0$ 或 $\mu$ 都好

---

### Q5
Linear schedule 下 $\beta_t \in [10^{-4}, 0.02]$。

如果 $\beta_{\max} = 0.5$：
- 单步加噪过强，$\bar\alpha_t$ 在很少几步内就接近 0
- 前向过程"信号丢失太快"，绝大部分 $t$ 都在"接近纯噪声"区域
- 训练时大部分 $t$ 的样本 $x_t \approx \epsilon$，网络很难学到有意义的去噪
- FID 显著恶化

---

### Q6
原因：
1. 每个 sample 只在一个随机 $t$ 上贡献 loss——大 batch 让所有 $t$ 都被"覆盖"得更均匀
2. Loss 的方差天然大（不同 $t$ 的难度差异），大 batch 减小梯度方差
3. EMA + AdamW 在大 batch 下更稳定
4. 大 batch 让 BatchNorm 替代品（GroupNorm）的统计更准确

---

## 第二部分：推导题

### Q7
**Base case ($t=1$)**：$x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1$，由 $\bar\alpha_1 = \alpha_1$ 命题成立。

**Inductive step**：设
$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} x_0 + \sqrt{1-\bar\alpha_{t-1}} \tilde\epsilon$$

代入 $x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_t$：

$$x_t = \sqrt{\alpha_t \bar\alpha_{t-1}} x_0 + \sqrt{\alpha_t(1-\bar\alpha_{t-1})} \tilde\epsilon + \sqrt{1-\alpha_t} \epsilon_t$$

注意 $\alpha_t \bar\alpha_{t-1} = \bar\alpha_t$。两个独立高斯之和：方差相加：
$$\alpha_t(1-\bar\alpha_{t-1}) + (1-\alpha_t) = 1 - \bar\alpha_t$$

所以 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$（$\epsilon \sim \mathcal{N}(0, I)$）。∎

（参考 derive_02 §2 完整版本）

---

### Q8
由贝叶斯：
$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

三个分子分母都是高斯。在指数里把所有含 $x_{t-1}$ 的项取出：

二次项系数：$-\frac{1}{2}\left[\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}}\right] \|x_{t-1}\|^2$

化简该系数 = $-\frac{1-\bar\alpha_t}{2\beta_t(1-\bar\alpha_{t-1})}$，即逆方差：
$$\tilde\beta_t = \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

一次项配方法给出：
$$\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t$$

（参考 derive_02 §3 完整推导）

---

### Q9
由 derive_02 §3.3，$\tilde\mu_t$ 用 $\epsilon$ 写成：
$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon\right)$$

$\mu_\theta$ 同样形式但 $\epsilon \to \epsilon_\theta$。所以：
$$\tilde\mu_t - \mu_\theta = \frac{1}{\sqrt{\alpha_t}} \cdot \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \cdot (\epsilon_\theta - \epsilon)$$

模长平方：
$$\|\tilde\mu_t - \mu_\theta\|^2 = \frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)} \cdot \|\epsilon - \epsilon_\theta\|^2$$

**系数**：$\dfrac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)}$，只依赖 $t$。

---

### Q10
(a) 由 $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$：
$$\nabla_{x_t} \log q(x_t|x_0) = -\frac{x_t - \sqrt{\bar\alpha_t} x_0}{1-\bar\alpha_t}$$

(b) 用 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$ 反推：
$$x_t - \sqrt{\bar\alpha_t} x_0 = \sqrt{1-\bar\alpha_t} \epsilon$$

代入：
$$\nabla_{x_t} \log q(x_t|x_0) = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

所以 $s_\theta = -\epsilon_\theta / \sqrt{1-\bar\alpha_t}$，**两者训练目标只差一个时间相关常数**。

(c) 转换系数：$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}$

---

## 第三部分：实验设计题

### Q11
(a) 4 个可能原因（典型答案）：
1. **训练步数不够**：DDPM 通常需要 200K+ steps
2. **EMA 没用**：DDPM 强依赖 EMA（exponential moving average of weights）
3. **学习率/warmup 不对**：典型 lr=1e-4 + 5K steps warmup
4. **数据增强不足**：CIFAR-10 训练应当用 random horizontal flip
5. **batch size 太小**：DDPM 论文用 128，小 batch 会让 loss 噪声过大

(b) 验证方法（举几例）：
- 训练步数：画 FID-vs-steps 曲线，看是否仍在下降
- EMA：在同一 checkpoint 上对比 use_ema 与否的 FID
- 学习率：在 1e-4 / 5e-5 / 2e-4 之间扫描，看哪个最稳定
- batch size：增大到 128 重训 50K steps 看趋势

(c) 评估端问题：
- **FID 计算样本数太少**（应 ≥ 10K）
- **训练集/测试集不一致**：FID 应该对 50K 测试集计算
- **Inception 模型不一致**：用错版本（如 PyTorch 实现 vs TF 原版）

---

### Q12
(a) Schedule 相关：
1. **改用 cosine schedule**：在 64×64+ 上经验上更好（中等 $t$ 区域信号保留更多）
2. **延长 T**：从 1000 → 4000 步，给模型更细的时间网格

(b) 训练目标：
- 引入 **learned variance + hybrid loss**（Improved DDPM）：在大图上 NLL 改进显著

(c) 答："64×64 不算'大图'，但 cosine 的边际收益已经显现。建议小心实验对比 linear vs cosine——在你的具体数据集上跑两次，看哪个 FID 低。Improved DDPM 论文里 64×64 ImageNet 上 cosine 比 linear 好约 0.3-0.5 FID，但低分辨率 CIFAR-10 上反而略差。所以不要照搬结论，要在你的设置下测。"

---

## 评分参考

- 完整答出 = 满分
- 思路对但细节缺失 = 70-80%
- 仅写公式不写推导步骤 = 50%
- 完全错或空 = 0
