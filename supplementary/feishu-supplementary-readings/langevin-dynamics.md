# 朗之万动力学理解

可以把这一节理解成一句话：

> **如果 score 告诉我们“往哪里走会更像真实数据”，那么 Langevin 动力学就是一种利用 score 从噪声中一步步走到真实数据分布的方法。**

核心公式是：

$$
x_{t+1}=x_t+\frac{\eta}{2}\nabla_x\log p(x_t)+\sqrt{\eta}\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$

---

## 1. 先看目标：我们想从 $p(x)$ 中采样

假设有一个复杂分布：

$$
p(x)
$$

比如真实图像分布。我们想生成样本，也就是得到：

$$
x \sim p(x)
$$

但问题是，真实数据分布通常很复杂，我们不能直接写出 $p(x)$，更不能直接采样。
Score-based model 的想法是：

> 不直接学 $p(x)$，而是学它的 score：

$$
s(x)=\nabla_x \log p(x)
$$

这个 score 告诉我们：当前位置 (x) 应该往哪里移动，才能更接近高概率区域。

------------------------------------------------------------------------

## 2. Langevin 动力学在做什么？

Langevin 迭代是：

$$
x_{t+1}
=x_t
+
\frac{\eta}{2}s(x_t)
+
\sqrt{\eta}\epsilon
$$

它包含两个部分。

----------------

### 第一项：朝高密度区域移动

$$
\frac{\eta}{2}s(x_t)
$$

因为：

$$
s(x_t)=\nabla_x \log p(x_t)
$$

所以这一项就是让 $x_t$ 沿着 $\log p(x)$ 上升最快的方向移动。
直觉上：

> 当前样本不太像真实数据，score 告诉它往哪里改，会变得更像真实数据。

比如一维标准高斯：

$$
p(x)=\mathcal{N}(0,1)
$$

它的 score 是：

$$
s(x)=-x
$$

所以 Langevin 的确定性部分是：

$$
x_{t+1}=x_t-\frac{\eta}{2}x_t
$$

这会把 $x_t$ 往 0 拉。因为标准高斯的高密度区域在 0 附近。
所以 drift 项的作用是：

> 把样本拉向高概率区域。

---

### 第二项：加入随机噪声

$$
\sqrt{\eta}\epsilon,
\quad
\epsilon\sim \mathcal{N}(0,I)
$$

这一项看起来像是在“捣乱”，但它非常重要。
如果只有第一项：

$$
x_{t+1}=x_t+\frac{\eta}{2}s(x_t)
$$

那么所有样本都会不断爬坡，最后集中到概率最大的地方，也就是 mode。
例如标准高斯中，所有点都会被拉到 0。
但真正的采样不是只找最大概率点。采样应该保留分布的形状：

- 有些样本在中心附近；
- 有些样本稍微偏远；
- 极少数样本在更远处。
  因此必须加入随机扰动，让样本不会全部坍缩到最高点。
  所以 noise 项的作用是：

> 保证样本能够探索整个分布，而不是只停在局部最高密度点。

---

## 2. 为什么 drift + noise 最后能得到 $p(x)$？

可以用一个直觉理解：
Langevin 动力学在两个力量之间达到平衡：

### 一边是 score 的吸引

$$
s(x)=\nabla_x \log p(x)
$$

它把样本推向高密度区域。

### 另一边是随机噪声的扩散

$$
\sqrt{\eta}\epsilon
$$

它让样本到处探索，避免全部挤在一起。
当这两者达到平衡时，样本的长期分布就是目标分布 (p(x))。
也就是说：

> score 负责“往真实数据区域靠近”，noise 负责“保持分布的多样性”。

---

## 3. 一个生活类比

可以把 $p(x)$ 想象成一个山谷地形：

- 高密度区域 = 更适合待的地方；
- score = 指向更高密度区域的方向；
- Langevin 采样 = 一群粒子在地形上移动。
  每一步粒子都会：

1. 顺着 score 往更高概率的区域移动；
2. 同时被随机风吹一下。
   如果只顺着 score 走，所有粒子都会挤到一个最高点。
   如果只有随机风，粒子会乱跑，没有目标。
   二者结合后，粒子最终会按照 $p(x)$ 的形状分布开来。

---

## 4. 为什么这对生成模型重要？

这一点非常关键：

> 如果我们能估计 $s(x)=\nabla_x\log p(x)$，就不一定需要直接知道 $p(x)$，也能从 $p(x)$ 中采样。

这就是 score-based generative model 的核心动机。
传统生成模型可能直接学习：

$$
p_\theta(x)
$$

但 score-based model 学的是：

$$
s_\theta(x)\approx \nabla_x\log p(x)
$$

然后通过 Langevin 动力学采样：

$$
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots
$$

最终让 $x_t$ 接近真实数据分布。

-------------------------------

## 5. 和 diffusion model 的关系

在 diffusion model 里，我们不是直接对干净数据分布 $p_{\text{data}}(x)$ 做 Langevin，而是对不同噪声水平下的分布 $p_t(x)$ 学 score：

$$
s_\theta(x,t)\approx \nabla_x \log p_t(x)
$$

其中 $t$ 表示噪声强度。
采样时，从纯噪声开始：

$$
x_T \sim \mathcal{N}(0,I)
$$

然后一步步用 score 指导去噪：

$$
x_T \rightarrow x_{T-1}\rightarrow \cdots \rightarrow x_0
$$

所以 diffusion 采样可以理解为：

> 在不同噪声层级下，不断使用 score 指导样本从噪声分布移动到数据分布。

---

## 6. 最关键的理解

Langevin 动力学说明了一件非常重要的事：

$$
\text{知道 } \nabla_x\log p(x)
\quad
\Rightarrow
\quad
\text{可以从 } p(x) \text{ 中采样}
$$

也就是说：

> **生成模型不一定要直接学概率密度 $p(x)$，只要学会 score，也可以生成样本。**

这就是 score-based model 和 diffusion model 的理论出发点。
