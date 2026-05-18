# Score Matching 解读

可以把这一段理解成一句话：

**Score matching 的核心问题是：我们想学“数据分布的梯度方向”，但真实分布密度不可见；所以要把一个看似无法监督的目标，改写成可以用样本训练的目标。**

-----------------------------------------------------------------------------------------------------------------------------------------------

## 1. 为什么要估计 score？

前面已经说过，score 是：

$$
s(x)=\nabla_x \log p(x)
$$

它表示：**在当前位置 (x)，往哪个方向走，数据概率密度会上升最快。**
如果我们知道真实 score，就可以用 Langevin Dynamics 从噪声中采样：

$$
x_{t+1}=x_t+\frac{\eta}{2}s(x_t)+\sqrt{\eta}\epsilon
$$

所以生成模型的一个思路就是：

> 不直接学 $p_{\text{data}}(x)$，而是学它的梯度场

$$
\nabla_x \log p_{\text{data}}(x)
$$

这就是 score-based generative model 的出发点。

----------------------------------------------

## 2. 朴素目标为什么不可行？

理想情况下，我们希望训练网络：

$$
s_\theta(x) \approx \nabla_x \log p_{\text{data}}(x)
$$

于是自然想到最小化：

$$
\mathbb{E}_{x \sim p_{\text{data}}}
\left||
s_\theta(x)-\nabla_x \log p_{\text{data}}(x)
\right||^2
$$

这看起来像普通监督学习：

$$
\text{预测值} - \text{真实标签}
$$

其中：

$$
s_\theta(x)
$$

是网络预测值，而

$$
\nabla_x \log p_{\text{data}}(x)
$$

是真实标签。
问题是：**我们没有这个真实标签。**
因为我们只有样本：

$$
x_1,x_2,\dots,x_N \sim p_{\text{data}}
$$

但不知道真实密度函数 $p_{\text{data}}(x)$，更不知道它的梯度。
所以朴素目标不可直接计算。

--------------------------

## 3. Hyvärinen Score Matching 做了什么？

Hyvärinen Score Matching 的关键贡献是：

> 虽然我们不知道真实 score，但可以把原目标等价改写成一个不含真实 score 的目标。

原目标是：

$$
\mathbb{E}_{p_{\text{data}}}
\left|
s_\theta(x)-\nabla_x \log p_{\text{data}}(x)
\right|^2
$$

展开平方：

$$
\mathbb{E}
\left[
|s_\theta(x)|^2
-2s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)
+|\nabla_x \log p_{\text{data}}(x)|^2
\right]
$$

最后一项：

$$
|\nabla_x \log p_{\text{data}}(x)|^2
$$

和模型参数 $\theta$ 无关，可以看作常数。
麻烦的是中间项：

$$
\mathbb{E}{p_{\text{data}}}
\left[
s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)
\right]
$$

Hyvärinen 的推导通过分部积分，把这个项变成了：

$$
-\mathbb{E}{p_{\text{data}}}
\left[
\text{tr}(\nabla_x s_\theta(x))
\right]
$$

于是目标就变成：

$$
\mathbb{E}_{x \sim p_{\text{data}}}
\left[
\text{tr}(\nabla_x s_\theta(x))
+
\frac{1}{2}|s_\theta(x)|^2
\right]
$$

这个目标里已经没有：

$$
\nabla_x \log p_{\text{data}}(x)
$$

所以它可以只用数据样本训练。

----------------------------

## 4. 但 Hyvärinen Score Matching 为什么不常直接用？

问题在这一项：

$$
\text{tr}(\nabla_x s_\theta(x))
$$

这里的 $\nabla_x s_\theta(x)$ 是网络输出对输入的雅可比矩阵。
假设图像维度是：

$$
x \in \mathbb{R}^{H \times W \times C}
$$

比如 $256 \times 256 \times 3$，维度接近 20 万。
那么 $s_\theta(x)$ 也是同样维度的向量。它对 $x$ 求导会产生一个巨大雅可比矩阵：

$$
\frac{\partial s_\theta(x)}{\partial x}
$$

而 trace 是这个矩阵对角线元素之和。
高维图像里，直接计算这个 trace 很慢，所以原始 score matching 理论漂亮，但实际训练大模型时不方便。

-------------------------------------------------------------------------------------------------

## 5. Denoising Score Matching 的直觉

DSM 的想法很聪明：

> 不直接学干净数据分布 $p_{\text{data}}$ 的 score，而是学加噪后分布的 score。

给干净样本 $x$ 加噪：

$$
\tilde{x}=x+\sigma \epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
$$

于是 $\tilde{x}$ 是一个带噪样本。
现在我们不直接估计：

$$
\nabla_x \log p_{\text{data}}(x)
$$

而是估计：

$$
\nabla_{\tilde{x}} \log p_\sigma(\tilde{x})
$$

也就是加噪分布的 score。

------------------------

## 6. 为什么 DSM 有监督信号？

关键在于：虽然整体的加噪分布 $p_\sigma(\tilde{x})$ 仍然不知道，但给定干净样本 $x$ 后，条件分布是已知的：

$$
p_\sigma(\tilde{x}|x)=\mathcal{N}(x,\sigma^2 I)
$$

也就是说：

$$
\tilde{x}
$$

是以 $x$ 为均值、$\sigma^2 I$ 为方差的高斯分布。
它的 log density 是：

$$
\log p_\sigma(\tilde{x}|x) =
-\frac{1}{2\sigma^2}|\tilde{x}-x|^2+\text{const}
$$

对 $\tilde{x}$ 求导：

$$
\nabla_{\tilde{x}} \log p_\sigma(\tilde{x}|x) =
-\frac{\tilde{x}-x}{\sigma^2}
$$

因为：

$$
\tilde{x}-x=\sigma \epsilon
$$

所以：

$$
-\frac{\tilde{x}-x}{\sigma^2} =
-\frac{\sigma \epsilon}{\sigma^2} =
-\frac{\epsilon}{\sigma}
$$

因此 DSM 的训练目标是：

$$
\mathcal{L}_{\mathrm{DSM}}=
\mathbb{E}_{x,\epsilon}
\left[
\left|
s_\theta(\tilde{x})
+
\frac{\epsilon}{\sigma}
\right|^2
\right]
$$

也就是说，网络要学习：

$$
s_\theta(\tilde{x}) \approx -\frac{\epsilon}{\sigma}
$$

这就是加噪样本的 score。

------------------------

## 7. DSM 和去噪有什么关系？

因为：

$$
s_\theta(\tilde{x}) \approx -\frac{\tilde{x}-x}{\sigma^2}
$$

所以可以反过来得到：

$$
x \approx \tilde{x}+\sigma^2 s_\theta(\tilde{x})
$$

这说明：

> 如果网络知道 score，它就知道如何从带噪样本 $\tilde{x}$ 回到干净样本 $x$。

所以 score matching 和 denoising 是同一件事的两个视角：

| 视角             | 网络学什么                  |
| -------------- | ---------------------- |
| Score matching | 学 $\nabla_x \log p(x)$ |
| Denoising      | 学如何把噪声去掉               |
| DDPM           | 学预测噪声 $\epsilon$       |


它们之间只差一个比例系数。

--------------------------

## 8. DSM 和 DDPM 的连接

在 DSM 中：

$$
s_\theta(\tilde{x}) \approx -\frac{\epsilon}{\sigma}
$$

所以如果网络预测的是噪声 $\epsilon_\theta(\tilde{x})$，则有：

$$
s_\theta(\tilde{x})
\approx
-\frac{1}{\sigma}\epsilon_\theta(\tilde{x})
$$

也就是说：

$$
\epsilon_\theta(\tilde{x})
\approx
-\sigma s_\theta(\tilde{x})
$$

这就是为什么说：

> DDPM 预测噪声，本质上等价于预测 score，只是差一个 $\sigma$ 系数。

在 DDPM 里，常见写法是：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
$$

这里的噪声标准差相当于：

$$
\sigma_t = \sqrt{1-\bar{\alpha}_t}
$$

所以 DDPM 训练 $\epsilon_\theta(x_t,t)$，本质上就是在不同噪声尺度下训练 score 网络。

------------------------------------------------------------------------------------

## 9. 为什么需要多尺度 score matching？

如果只用一个固定噪声尺度 $\sigma$，会有问题。

### 情况一：$\sigma$ 太小

噪声太小，带噪样本几乎贴着数据流形。
比如真实图像都在一个很薄的低维流形上，$\sigma$ 很小时，训练样本只覆盖流形附近。
那么在远离数据流形的地方，网络几乎没有训练信号。
但采样时，一开始通常从纯噪声开始，位置可能离真实数据很远。
所以小噪声 score 对采样初期帮助不大。

-------------------------------------

### 情况二：$\sigma$ 太大

噪声太大，数据结构被严重破坏。
比如一张猫图加了很大噪声后，看起来接近随机噪声。
这时候网络能学到大的轮廓方向，但难以恢复细节。
所以大噪声有利于全局探索，小噪声有利于细节恢复。

------------------------------------------------

## 10. NCSN 的核心思想

NCSN 使用多个噪声尺度：

$$
\sigma_1 > \sigma_2 > \cdots > \sigma_L
$$

训练条件 score 网络：

$$
s_\theta(x,\sigma)
$$

让它在不同噪声尺度下都能预测 score。
直观理解：
| 噪声尺度       | 学到什么       |
| ---------- | ---------- |
| 大 $\sigma$ | 粗粒度结构，全局方向 |
| 中 $\sigma$ | 物体布局，主要形状  |
| 小 $\sigma$ | 局部纹理，细节恢复  |


采样时，从大噪声开始：

$$
x_L \sim \mathcal{N}(0,\sigma_L^2I)
$$

然后逐步降低噪声尺度：

$$
\sigma_1 \rightarrow \sigma_2 \rightarrow \cdots \rightarrow \sigma_L
$$

每个尺度下做若干步 Langevin Dynamics。
直觉上就是：

> 先在模糊的大尺度分布里找到大致图像结构，再逐步细化成真实图像。

---

## 11. NCSN 和 DDPM 的关系

NCSN 的流程是：

$$
\text{大噪声} \rightarrow \text{中噪声} \rightarrow \text{小噪声} \rightarrow \text{干净数据}
$$

DDPM 的反向过程也是：

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
$$

也就是：

$$
\text{纯噪声} \rightarrow \text{逐步去噪} \rightarrow \text{干净图像}
$$

所以两者本质非常接近。
区别在于：
| 方法        | 表达方式                 |
| --------- | -------------------- |
| NCSN      | 显式训练不同噪声尺度下的 score   |
| DDPM      | 训练不同时间步 $t$ 下的噪声预测网络 |
| Score SDE | 用连续时间 SDE 统一两者       |

因此可以把 DDPM 理解为一种特殊的多尺度 denoising score matching。

-----------------------------------------------------------------

## 12. 最核心的理解

这一节可以压缩成三句话：
**第一，score-based 生成模型想学的是：**

$$
\nabla_x \log p_{\text{data}}(x)
$$

也就是数据分布的梯度场。
**第二，真实 score 不可见，所以 Hyvärinen Score Matching 通过数学等价变换，绕开真实 score；DSM 则通过人为加噪，让监督信号变成已知的高斯噪声。**
**第三，DDPM 预测噪声 (\epsilon)，本质上就是在不同噪声尺度下预测 score，因此 DDPM、DSM、NCSN 都可以看作同一条思想线的不同形式。**
最终关系是：

$$
\boxed{
\text{预测噪声}
\Longleftrightarrow
\text{预测 score}
\Longleftrightarrow
\text{逐步去噪生成}
}
$$

这就是从 score matching 到 diffusion model 的关键桥梁。
