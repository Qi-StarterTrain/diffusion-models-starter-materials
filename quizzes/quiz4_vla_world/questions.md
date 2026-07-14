# Quiz 4：World Models + VLA / Embodied（W15-W16）

> **时长**：90 分钟
> **总分**：100 分
> **题型**：概念题 30 分 + 推导题 40 分 + 实验设计题 30 分
> **范围**：L16-L17、paper notes 18-19、derive_07-10（FM 在 VLA 的应用）

---

## 一、概念题（30 分）

### 1（5 分）Video generation 与 world model 的核心区别是什么？为什么 Sora 严格说不是 world model？

### 2（5 分）4 个 world model 能力等级（L1-L4）分别是什么？给每个等级一个代表模型。

### 3（5 分）Pi-0 用 Flow Matching 而不是 DDPM 作 action head，主要工程原因有哪些？（至少列 3 点）

### 4（5 分）OpenVLA 的 action 用 discrete token，Pi-0 的 action 是 continuous trajectory via FM。这两种选择各自的优劣是什么？

### 5（5 分）Action chunking 是什么？为什么 chunk size 通常 16-50 而不是单步？

### 6（5 分）Diffusion Policy 比 Behavior Cloning (MSE) 在 robotics 上的核心优势是什么？给一个具体场景示例。

---

## 二、推导题（40 分）

### 7（10 分）Flow Matching 在 action 生成上的训练目标

设：
- $a_{1:H} \in \mathbb{R}^{H \times A}$：未来 $H$ 步 action chunk
- vision feature $v \in \mathbb{R}^{D_v}$
- language feature $l \in \mathbb{R}^{D_l}$
- noise $\epsilon \sim \mathcal{N}(0, I)$，shape $H \times A$
- $t \in [0, 1]$

(a)（3 分）写出 linear-path action FM 的训练 loss（参数化为 $v_\theta$）。

(b)（4 分）推断时，给定 obs (vision + language)，写出从 noise 到 action chunk 的 Euler ODE 采样代码（伪代码 OK）。

(c)（3 分）若 50 Hz 控制频率要求，single forward pass 约 4ms，最多多少步 FM 采样可行？

---

### 8（10 分）Compound error 在 autoregressive world model 中的累积

设每步预测正确概率为 $p = 0.99$（即每步独立错误率 $1 - p = 0.01$）。

(a)（3 分）假设错误独立，预测 100 步后**全程**正确的概率是多少？

(b)（4 分）实际上误差会**累积**（前步错导致后续 conditional input 错），假设错误率每步增 10%。写出第 $k$ 步的预期错误率公式，并估算第 50 步与第 100 步的错误率。

(c)（3 分）这说明什么？为什么 Sora 60 秒视频和真正的 world model 之间存在巨大 gap？

---

### 9（10 分）OpenVLA 的 action tokenization 量化误差

OpenVLA 把 continuous action（7 dim）的每个维度量化为 256 bins，bin 边界按 quantile 划分。

(a)（3 分）假设 action 在 [-1, 1] 均匀分布，bin 是等宽，每 bin 的量化误差 RMS 是多少？

(b)（4 分）实际上 action distribution 不均匀（少数 bin 占大部分 mass）。Quantile-based binning 在这种情况下相比等宽 binning 有何优势？写出数学论证或直觉。

(c)（3 分）若想把 256 bins 降到 64 bins（节省 token 数 / inference），评估的 RMS 误差增加多少？这种 trade-off 对实时控制可接受吗？

---

### 10（10 分）Vision encoder 对 VLA 性能的影响

设 VLA 用 ImageNet-pretrained ResNet18 作 vision encoder（特征维 512）。

(a)（4 分）解释为什么使用预训练 vision encoder 比从头训对 small-data VLA 更有效。从信息论或表示学习角度论证。

(b)（3 分）OpenVLA 用了 **DINOv2 + SigLIP** 两个 encoder concat。直觉上这两个 encoder 互补在哪里？

(c)（3 分）训练 VLA 时是否要 fine-tune vision encoder？支持 / 反对的论据各写一个。

---

## 三、实验设计题（30 分）

### 11（15 分）设计一个 VLA 的"sim-to-real gap"评估

你训了一个 VLA 在仿真环境中（Mujoco）成功率 90%。部署到真机后发现成功率仅 30%。请设计：

(a)（5 分）3 个可能原因的假设（hypothesis）

(b)（5 分）为每个假设设计一个**最小可验证**的实验（控制变量 + 评估指标）

(c)（5 分）若所有 3 个假设都成立，你提出的解决方案 priority order 是什么？

---

### 12（15 分）国自然青基级实验设计：基于 FM 的实时 VLA

题目：**"面向高频控制的 Flow Matching 视觉-语言-动作模型"**

设计一个 3-experiment 路线，能在 1 年内完成并发表 ICRA/CoRL：

(a)（5 分）Experiment 1（最小可行验证）：在简单 task 上证明 FM 比 DDPM 快 N×，质量不降

(b)（5 分）Experiment 2（生态适配）：把现有 VLA backbone (OpenVLA) 改造为 FM action head，做 ablation

(c)（5 分）Experiment 3（gap to deployment）：在真机上测试，分析数据效率与失败模式

每 experiment 要写：**目标**、**setup**、**评估**、**预期结果**。

---

## 评分标准

- 概念题：答到关键点得分
- 推导题：每步骤 / 量纲 / 结论分别得分
- 实验设计题：合理性 > 完美性。**自我批判**与**对照实验**是高分关键
- 创新答案（已超出课程内容）：教学评语 + 满分

---

## 答案见 `answer_key.md`（请不要先翻！）

> **此 Quiz 完成 = 你已具备 VLA 研究者基础。下一步是动手做。**
