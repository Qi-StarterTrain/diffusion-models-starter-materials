# Quiz 3：DiT / Flow Matching / Consistency / Video（W10-W14）

> **时长**：90 分钟
> **总分**：100 分
> **题型**：概念题 30 分 + 推导题 40 分 + 实验设计题 30 分
> **范围**：L11-L15、derive_07-10、paper notes 12-17

---

## 一、概念题（30 分，每题 5 分，共 6 题）

### 1. AdaLN-Zero 的"zero init"指的是什么？为什么这是必要的？

### 2. 简述 Flow Matching 与 DDPM 的本质区别。在工程上哪一个更简洁？为什么？

### 3. Consistency Model 与 GAN 都能"一步生成"。它们的训练机制有何不同？

### 4. 视频扩散模型用 "spatial + temporal factorized attention"，列出这种 factorization 的计算复杂度优势与表达能力代价。

### 5. Sora 的 "spacetime patches" 是什么？它如何让模型支持任意分辨率与时长？

### 6. ControlNet 中 zero convolution 与 LoRA 中 B=0 init 的设计哲学共通点是什么？分别保护了什么？

---

## 二、推导题（40 分）

### 7（10 分）Linear-path Flow Matching loss 推导

设 $x_0 \sim \mathcal{N}(0, I)$，$x_1 \sim p_{\text{data}}$。定义 linear path：
$$x_t = (1-t) x_0 + t x_1, \quad t \in [0, 1]$$

(a)（3 分）证明 $u_t(x_t | x_1) = x_1 - x_0$。

(b)（4 分）写出 Conditional Flow Matching loss $\mathcal{L}_{\text{CFM}}$ 的具体表达。

(c)（3 分）解释为什么 $\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$（直觉论证即可，不需要严格证明）。

---

### 8（10 分）Consistency Model 的 EDM-style 参数化

Consistency function 形如：
$$f_\theta(x, t) = c_{\text{skip}}(t) \cdot x + c_{\text{out}}(t) \cdot F_\theta\bigl(c_{\text{in}}(t) \cdot x, c_{\text{noise}}(t)\bigr)$$

(a)（4 分）说明每个 $c$ 函数的作用，特别是 $c_{\text{skip}}, c_{\text{out}}$ 在 $t = \epsilon$ 处必须满足什么条件？

(b)（3 分）写出 CD（Consistency Distillation）loss 的形式。说明 EMA teacher $\theta^-$ 的作用。

(c)（3 分）CT（Consistency Training）不用 teacher，依赖什么 trick？

---

### 9（10 分）DiT 计算复杂度

假设 DiT 输入 $z \in \mathbb{R}^{4 \times 32 \times 32}$，patch size $p = 2$，embedding dim $d = 1152$，深度 $L = 28$，head 数 16。

(a)（3 分）算出 token 数 $N$ 与每个 transformer block 的 self-attention FLOPs（忽略 softmax 与 mask）。

(b)（3 分）算出每个 block 的 FFN（GeLU，hidden 4d）FLOPs。

(c)（4 分）总训练步 FLOPs（forward + backward）若 batch=256 跑 1M 步是多少？相对 SD UNet（约 0.5 TFLOPs per image at 256×256）多多少倍？

---

### 10（10 分）视频 Attention factorization

假设 video latent shape $(T, H, W) = (32, 32, 32)$，token 数 $N = T \cdot H \cdot W = 32768$。

(a)（3 分）计算"完整 3D self-attention" 的 attention matrix 大小（以 GB 为单位，fp16）。

(b)（3 分）若改为 "spatial + temporal factorized"：spatial 在每帧内做 $(HW)^2$，temporal 在每空间位置做 $T^2$。计算总 attention matrix 大小。

(c)（4 分）若再加 window attention（window 大小 $w_t \times w_h \times w_w = 4 \times 8 \times 8$），attention matrix 总大小？比 full 3D 节省多少倍？

---

## 三、实验设计题（30 分）

### 11（15 分）FM vs DDPM 公平对比

你想发表一篇 "Flow Matching 在 small data regime 是否真的优于 DDPM"。请设计：

(a)（5 分）实验设置（数据集、模型、训练规模）：要保证公平对比。

(b)（5 分）评估指标（至少 3 种），各自适合 capture 什么性质。

(c)（5 分）你预期会观察到什么？如何反驳"Flow Matching 一定优于 DDPM" 这个 hype？

---

### 12（15 分）VLA 中的 action diffusion 设计

你要为一个特定 robot（如 7-DoF arm）设计 action head：

(a)（5 分）action 维度与 chunking size 怎么选？写出你的设计 + reasoning。

(b)（5 分）选 DDPM、Flow Matching、Consistency Model 中哪个？说明你的 trade-off 分析（控制频率、训练成本、质量）。

(c)（5 分）如何评估这个 action head 的性能？至少 3 个评估指标 + 1 个 ablation 实验。

---

## 评分标准

- **概念题**：答出关键词得分；额外补充正确观点最多 +1
- **推导题**：每步骤清晰、每个量量纲对、最终结果接近正确即可得部分分
- **实验设计题**：设计合理、有自我批判（"我不能确定 X，所以加 Y 对照"）得高分

---

## 答案见 `answer_key.md`（请不要先翻！）
