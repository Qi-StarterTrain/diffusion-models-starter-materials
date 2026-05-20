# Quiz 1: DDPM 基础（W1-W4）

> **时长建议**：90 分钟
> **形式**：闭卷，可携带一页 A4 手写公式
> **范围**：L01-L05 + derive_01 至 derive_03

---

## 第一部分：概念题（每题 5 分，共 30 分）

### Q1
请用一句话区分以下三个概念：
- $q(x_t | x_0)$
- $q(x_t | x_{t-1})$
- $q(x_{t-1} | x_t, x_0)$

DDPM 训练用到了哪一个？为什么？

---

### Q2
DDPM 训练目标 $\mathcal{L}_{\text{simple}}$ 与严格 ELBO 推导出来的损失差什么？为什么 DDPM 用前者？

---

### Q3
判断对错并说明理由：
> 训练好的 DDPM 模型每次采样输出固定（确定性的）。

---

### Q4
为什么 DDPM 网络选择预测 $\epsilon$ 而非 $x_0$ 或 $x_{t-1}$ 或 $\mu$？至少给出两个原因。

---

### Q5
DDPM 中 $\beta_t$ 的取值范围（在 linear schedule 下）。如果把 $\beta_{\max}$ 改成 0.5（远高于标准），训练会出什么问题？

---

### Q6
解释为什么扩散模型用大 batch（如 256-1024）效果好，而不是像 GAN 那样小 batch。

---

## 第二部分：推导题（每题 10 分，共 40 分）

### Q7
从 $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$ 出发，**用归纳法**证明：
$$q(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$$

要求：写出归纳假设、归纳步骤、关键技巧（独立高斯之和）。

---

### Q8
用贝叶斯定理 + 高斯运算推出 $q(x_{t-1} | x_t, x_0)$ 的均值 $\tilde\mu_t$ 与方差 $\tilde\beta_t$。

允许直接用 $q(x_t | x_0)$ 的结论（即 Q7 的结果）。

---

### Q9
DDPM 网络参数化为 $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta)$。

证明 $\|\tilde\mu_t - \mu_\theta\|^2$ 等于 $\|\epsilon - \epsilon_\theta\|^2$ 乘上一个**只依赖于 $t$ 的系数**，给出这个系数的表达式。

---

### Q10
解释 noise prediction 与 score matching 的等价性。具体地：

(a) 写出 $\nabla_{x_t} \log q(x_t | x_0)$ 的解析表达式。

(b) 用上式说明：训练 $\epsilon_\theta(x_t, t) \approx \epsilon$ 等价于训练 $s_\theta(x_t, t) \approx \nabla_{x_t} \log p_t(x_t)$。

(c) 给出两者之间的转换系数。

---

## 第三部分：实验设计题（每题 15 分，共 30 分）

### Q11

你训练了一个 DDPM on CIFAR-10，跑出了 FID = 8.5，但你的 baseline 论文 FID = 3.5。你怀疑训练不充分。

(a) 列出**至少 4 个可能原因**（按你认为可能性从高到低）。

(b) 对每个原因，描述你会如何**实验性地验证**它（具体动作）。

(c) 如果排除完训练问题，FID 仍然在 8.5 附近，可能是评估端出了什么问题？

---

### Q12

你的同事用 `linear schedule` 训练 32×32 RGB 图，FID=3.5；现在他想换 64×64 训练，但 FID 跳到 5.8。

(a) 提出**两个**与 schedule 设计相关的修改建议，分别说明动机。

(b) 提出**一个**与训练目标相关的修改建议（Hint: Improved DDPM）。

(c) 你建议他用 cosine schedule，但他说"看了 Improved DDPM 论文，cosine 只在大图上才好，64×64 算大图吗？"——如何回答？

---

## 提交格式

- 推导题手写或 LaTeX 均可（拍照清晰、字迹工整）
- 实验设计题用文字回答即可
- 每题独立编号，不要省略题号

---

> **完成 Quiz 1 后，进入 W5（Improved DDPM）**。建议把不会的题目当作"回滚知识点"信号——通常说明前面某讲没消化透。
