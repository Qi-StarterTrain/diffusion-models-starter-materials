# 高斯 Score 推导：为什么预测噪声等价于预测 Score

这份补充阅读解释扩散模型中一个关键公式：

$$
\nabla_{x_t}\log q(x_t|x_0)
=
-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}
$$

它说明：

**DDPM 中预测噪声 $\epsilon$，本质上等价于预测 score $\nabla_{x_t}\log q(x_t|x_0)$，只差一个时间相关系数。**

## 1. 先从一般高斯分布开始

假设：

$$
p(x)=\mathcal N(\mu,\sigma^2 I)
$$

其中 $x\in\mathbb R^d$。

高斯密度是：

$$
p(x)
=
\frac{1}{(2\pi\sigma^2)^{d/2}}
\exp\left(
-\frac{1}{2\sigma^2}\|x-\mu\|^2
\right)
$$

对它取 log：

$$
\log p(x)
=
-\frac{d}{2}\log(2\pi\sigma^2)
-
\frac{1}{2\sigma^2}\|x-\mu\|^2
$$

第一项：

$$
-\frac{d}{2}\log(2\pi\sigma^2)
$$

与 $x$ 无关，所以对 $x$ 求梯度时为 $0$。

真正需要求导的是第二项：

$$
-\frac{1}{2\sigma^2}\|x-\mu\|^2
$$

## 2. 对二次项求梯度

注意：

$$
\|x-\mu\|^2
=
(x-\mu)^\top(x-\mu)
$$

对 $x$ 求梯度：

$$
\nabla_x \|x-\mu\|^2
=
2(x-\mu)
$$

因此：

$$
\nabla_x\log p(x)
=
-\frac{1}{2\sigma^2}\cdot 2(x-\mu)
$$

得到：

$$
\boxed{
\nabla_x\log p(x)
=
-\frac{x-\mu}{\sigma^2}
}
$$

这就是高斯分布的 score。

## 3. 这个公式的几何含义

score 是：

$$
\nabla_x\log p(x)
$$

它指向密度上升最快的方向。

对于高斯：

$$
\nabla_x\log p(x)
=
-\frac{x-\mu}{\sigma^2}
=
\frac{\mu-x}{\sigma^2}
$$

所以 score 指向均值 $\mu$。

直觉上：

- 如果 $x$ 在均值右边，score 指向左边；
- 如果 $x$ 在均值左边，score 指向右边；
- 离均值越远，$|x-\mu|$ 越大，score 的模长越大；
- $\sigma^2$ 越大，分布越平，score 越弱。

因此高斯 score 可以理解为：

> 把样本拉回高密度中心的方向。

## 4. 代入 DDPM 的 $q(x_t|x_0)$

DDPM 前向加噪的闭合形式是：

$$
q(x_t|x_0)
=
\mathcal N
\left(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right)
$$

也可以写成重参数化形式：

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
$$

和一般高斯：

$$
\mathcal N(\mu,\sigma^2I)
$$

对比可得：

$$
\mu_t
=
\sqrt{\bar\alpha_t}x_0
$$

以及：

$$
\sigma_t^2
=
1-\bar\alpha_t
$$

所以根据高斯 score 公式：

$$
\nabla_{x_t}\log q(x_t|x_0)
=
-\frac{x_t-\mu_t}{\sigma_t^2}
$$

代入 $\mu_t$ 和 $\sigma_t^2$：

$$
\nabla_{x_t}\log q(x_t|x_0)
=
-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}
$$

## 5. 用噪声 $\epsilon$ 化简

由前向加噪公式：

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

移项得到：

$$
x_t-\sqrt{\bar\alpha_t}x_0
=
\sqrt{1-\bar\alpha_t}\epsilon
$$

代回 score：

$$
\nabla_{x_t}\log q(x_t|x_0)
=
-\frac{\sqrt{1-\bar\alpha_t}\epsilon}{1-\bar\alpha_t}
$$

约掉一个 $\sqrt{1-\bar\alpha_t}$：

$$
\boxed{
\nabla_{x_t}\log q(x_t|x_0)
=
-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}
}
$$

这就是目标公式。

## 6. 为什么说预测噪声等价于预测 score

上式可以改写成：

$$
\epsilon
=
-\sqrt{1-\bar\alpha_t}
\nabla_{x_t}\log q(x_t|x_0)
$$

也就是说，如果网络能预测噪声：

$$
\epsilon_\theta(x_t,t)\approx \epsilon
$$

那么就可以得到对应的 score 估计：

$$
s_\theta(x_t,t)
\approx
-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}
$$

反过来，如果网络能预测 score，也能乘上时间相关系数得到噪声。

所以：

$$
\boxed{
\text{预测噪声}
\Longleftrightarrow
\text{预测 score}
}
$$

两者只差一个依赖时间 $t$ 的缩放因子。

## 7. 注意：这里是条件 score

上面推导的是：

$$
\nabla_{x_t}\log q(x_t|x_0)
$$

也就是给定干净样本 $x_0$ 后，带噪样本 $x_t$ 的条件分布 score。

而反向 SDE / probability flow ODE 里真正需要的是边缘分布的 score：

$$
\nabla_{x_t}\log q_t(x_t)
$$

其中：

$$
q_t(x_t)
=
\int q(x_t|x_0)p_{\mathrm{data}}(x_0)\,dx_0
$$

训练时用 denoising score matching 的结论，把条件 score 作为监督信号来学习边缘 score。直觉上，网络看到大量 $(x_0,\epsilon,t)$ 样本后，会学到在每个噪声水平下如何从 $x_t$ 指向更高概率的数据区域。

## 8. 一句话总结

DDPM 的前向分布 $q(x_t|x_0)$ 是高斯，因此它的 score 可以直接解析求出：

$$
\nabla_{x_t}\log q(x_t|x_0)
=
-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}
=
-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}
$$

所以训练网络预测噪声 $\epsilon$，等价于训练它预测当前噪声水平下的 score，只是参数化方式更方便、更稳定。
