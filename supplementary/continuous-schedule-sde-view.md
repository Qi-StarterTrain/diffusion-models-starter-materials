# Continuous Schedule 与 SDE 视角

在 DDPM 的课件里，`continuous schedule` 指的是：**不再把加噪过程理解为有限个离散时间步上的噪声注入，而是把它看成一个连续时间的随机过程。**

也就是说，原来我们写的是：

$$
t=1,2,\dots,T
$$

并为每个时间步指定一个：

$$
\beta_t
$$

当 $T$ 越来越大、每一步越来越小时，这组离散参数就会逼近一条连续函数：

$$
\beta(t), \qquad t \in [0,1]
$$

这就是 continuous schedule 的核心思想。

## 1. 从离散 DDPM 出发

DDPM 的 forward process 定义为：

$$
q(x_t \mid x_{t-1}) = \mathcal N \left(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I\right)
$$

等价地，

$$
x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t,\qquad \epsilon_t\sim\mathcal N(0,I)
$$

这里的 $\beta_t$ 表示第 $t$ 步加入多少噪声，因此它被称为 noise schedule。

在离散模型里，schedule 是一个长度为 $T$ 的序列：

$$
\{\beta_1,\beta_2,\dots,\beta_T\}
$$

而 continuous schedule 则把它升级成一条连续曲线：

$$
\beta(t)
$$

## 2. 为什么 $T \to \infty$ 会走向 SDE

如果每一步都非常小，也就是：

$$
\beta_t \ll 1
$$

那么可以做一阶近似：

$$
\sqrt{1-\beta_t}\approx 1-\frac12\beta_t
$$

代回离散更新公式：

$$
x_t \approx \left(1-\frac12\beta_t\right)x_{t-1}+\sqrt{\beta_t}\epsilon_t
$$

移项得到：

$$
x_t-x_{t-1}\approx -\frac12\beta_t x_{t-1}+\sqrt{\beta_t}\epsilon_t
$$

现在把每一步看成长度为 $\Delta t$ 的连续时间增量，并写成：

$$
\beta_t=\beta(t)\Delta t
$$

则上式变成：

$$
x_{t+\Delta t}-x_t\approx -\frac12\beta(t)x_t\Delta t+\sqrt{\beta(t)\Delta t}\,\epsilon_t
$$

当：

$$
\Delta t\to 0
$$

时，$\sqrt{\Delta t}\epsilon_t$ 会对应到布朗运动增量 $dW_t$，于是极限过程变成：

$$
\boxed{dx=-\frac12\beta(t)x\,dt+\sqrt{\beta(t)}\,dW_t}
$$

这就是 DDPM forward process 在连续时间下的 SDE 形式，通常叫做 **variance-preserving SDE (VP-SDE)**。

## 3. 这个 SDE 在表达什么

这个方程里有两部分：

第一部分是漂移项：

$$
-\frac12\beta(t)x\,dt
$$

它表示当前样本会被持续往 0 拉回去，也就是原始信号会逐渐衰减。

第二部分是扩散项：

$$
\sqrt{\beta(t)}\,dW_t
$$

它表示系统会不断注入高斯噪声，而且注入强度由 $\beta(t)$ 控制。

所以 continuous schedule 的物理图像可以理解为：

$$
\text{信号持续衰减} + \text{噪声持续注入}
$$

离散 DDPM 是“每一步加一点噪声”，连续 SDE 是“每一瞬间都在加极小的噪声”。

## 4. 连续时间下的 $\bar\alpha(t)$

离散 DDPM 中有一个非常重要的量：

$$
\bar\alpha_t=\prod_{s=1}^t(1-\beta_s)
$$

它表示经过很多步以后，还剩下多少信号比例。

在 continuous schedule 里，这个离散乘积会变成指数形式。因为当每一步都很小时：

$$
\log(1-\beta_s)\approx -\beta_s
$$

因此：

$$
\log \bar\alpha_t
=
\sum_{s=1}^t \log(1-\beta_s)
\approx
-\sum_{s=1}^t \beta_s
$$

若再把求和变成积分：

$$
\sum_{s=1}^t \beta_s \approx \int_0^t \beta(\tau)\,d\tau
$$

就得到：

$$
\boxed{
\bar\alpha(t)=\exp\left(-\int_0^t \beta(\tau)\,d\tau\right)
}
$$

这说明 continuous schedule 下，真正控制信号保留比例的，不再是离散乘积，而是：

$$
\int_0^t \beta(\tau)\,d\tau
$$

也就是累计噪声强度。

## 5. 闭式公式在连续时间下仍然成立

离散情形中，我们有：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
$$

在连续时间下，对应形式变成：

$$
\boxed{
x(t)=\sqrt{\bar\alpha(t)}\,x_0+\sqrt{1-\bar\alpha(t)}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
}
$$

其中：

$$
\bar\alpha(t)=\exp\left(-\int_0^t \beta(\tau)\,d\tau\right)
$$

这说明离散 DDPM 和连续 SDE 在本质上描述的是同一个加噪过程，只是一个用有限步近似，一个用连续时间表达。

## 6. 为什么这件事重要

continuous schedule 的意义不只是“把符号写得更高级”，而是它把 DDPM 和后续的 score-based generative modeling 连起来了。

从这个视角看：

- DDPM 中的 $\beta_t$ 序列，本质上是在离散采样一条连续函数 $\beta(t)$；
- 设计 noise schedule，本质上是在设计连续时间里的噪声注入速率；
- 当 $T$ 足够大时，离散扩散模型可以看成连续 SDE 的数值离散化。

这也是为什么后续很多论文会直接讨论：

- continuous-time noise schedule
- SNR parameterization
- reverse-time SDE
- probability flow ODE

因为在连续时间框架下，这些对象都更自然。

## 7. 和课件那句话对应起来

课件里写：

> Continuous schedule: $T \to \infty$ 极限下的 SDE 视角

这句话可以压缩理解成三层意思：

1. 原本的 DDPM 是一个离散马尔可夫链。
2. 当时间步无限细时，它会收敛到一个连续随机过程。
3. 这个连续随机过程可以用 SDE 来描述，而 schedule 也从 $\beta_t$ 序列变成 $\beta(t)$ 函数。

## 8. 一句话总结

**continuous schedule 的本质是：把“第 1 步、第 2 步、...、第 T 步”的加噪序列，提升为“在连续时间上按速率 $\beta(t)$ 不断衰减信号并注入噪声”的 SDE 过程。**

如果只记一个公式，就记：

$$
\boxed{dx=-\frac12\beta(t)x\,dt+\sqrt{\beta(t)}\,dW_t}
$$

以及：

$$
\boxed{\bar\alpha(t)=\exp\left(-\int_0^t \beta(\tau)\,d\tau\right)}
$$

前者是连续加噪动力学，后者是连续时间下的信号保留比例。
