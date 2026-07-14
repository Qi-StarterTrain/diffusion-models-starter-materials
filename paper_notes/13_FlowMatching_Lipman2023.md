# 论文导读 13: Flow Matching (Lipman et al., ICLR 2023)

**标题**：Flow Matching for Generative Modeling
**作者**：Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matthew Le (Meta AI)
**核心地位**：⭐⭐⭐⭐⭐ 后扩散时代的新范式，SD 3 / Pi-0 的数学基础

---

## 一、为什么必读

- 提出一个比 DDPM 更简洁的训练框架
- ODE-only（无 SDE），数学更干净
- 实证上少步采样质量优于 DDPM
- SD 3、Pi-0、AuraFlow 都基于此

---

## 二、阅读路线图

| 章节 | 优先级 |
|------|--------|
| §1 Intro | ✅ |
| §2 Background | ✅ |
| **§3 Flow Matching** | 🔴 核心 |
| §3.1 FM loss | 🔴 核心 |
| §3.2 Conditional FM | 🔴 核心 |
| §3.3 Conditional probability paths | 🔴 核心 |
| §4 Experiments | ✅ |

---

## 三、核心公式

### FM loss（不可计算）
$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x \sim p_t}\left[\|v_\theta(x, t) - u_t(x)\|^2\right]$$

### CFM loss（可计算）
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_1, x \sim p_t(\cdot|x_1)}\left[\|v_\theta(x, t) - u_t(x|x_1)\|^2\right]$$

### 等价性
$$\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$$

详见 derive_07。

---

## 四、Linear Path（最常用）

$x_t = (1-t) x_0 + t x_1$，对应 $u_t(x_t | x_1) = x_1 - x_0$。

训练 loss：
$$\mathcal{L} = \mathbb{E}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

—— 极简。

---

## 五、与 DDPM 的关系

DDPM 用 $\bar\alpha_t$ schedule（凸函数），FM 用 $t$（线性）。

差异：**轨迹 straightness**。
- DDPM：弯曲（特别在大 t 处）
- FM (linear)：直

→ FM 在少步采样下质量更高（直线积分更精确）。

---

## 六、与 EDM 的关系

EDM 是 DDPM 的"理论 cleanup"，FM 是"完全 rewrite"。
- EDM 仍基于 SDE
- FM 抛弃 SDE

两者都好，但 FM 的简洁性让它成为新主流。

---

## 七、实验结果

ImageNet 64（log-likelihood）：
- DDPM: 3.39 nats
- EDM: 3.18 nats
- **FM (OT)**: **3.31 nats**（相当）

CIFAR-10 NFE-FID Pareto：FM 在 10-20 NFE 区间优于 DDPM。

---

## 八、常被误读

### 1. "FM 是 DDPM 的特例？"
**反过来**。DDPM 可以重写成 FM 的一种特例（具体 schedule）。FM 是更一般的框架。

### 2. "Linear path 是 OT 路径？"
**部分是**。对高斯先验 + L2 cost 的 OT 是 linear；其他 cost 不是。但实践用 linear 已足够。

### 3. "CFM 与 FM 数学等价？"
**梯度等价**，不是 loss 等价。两个 loss 的数值不同（常数差），但对 $\theta$ 的偏导相同。

---

## 九、与 SD 3 的关系

SD 3 (Esser et al., 2024) 用 rectified flow + DiT，最重要的改进：
- Logit-normal t 采样（替代均匀）
- MMDiT 架构（替代 cross-attention DiT）

但**核心训练 loss 来自这篇 FM 论文**。

---

## 十、思考题

1. 推导 linear path 的 $u_t(x|x_1) = x_1 - x_0$（参考 derive_07）
2. 在 2D 数据上比较 OT path 与 linear path 的训练 loss 曲线
3. 用 FM 训练再做 reflow（Liu 2022），观察 1-step 质量改善
4. 思考：FM 可以做 "score distillation"吗？怎么把 SD 1.5 (DDPM) 转成 FM？

---

## 十一、引用

```
@inproceedings{lipman2023flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matthew},
  booktitle={ICLR},
  year={2023}
}
```
