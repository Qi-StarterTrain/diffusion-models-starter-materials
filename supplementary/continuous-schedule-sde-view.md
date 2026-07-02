# Continuous Schedule 与 SDE 视角：如何理解 $\beta_t = \beta(s)\Delta t$

这份补充阅读对应 `derive_04_score_sde.md` 的 1.1 小节，专门解释一个很容易让人卡住的式子：

$$
\beta_t = \beta(s)\Delta t,\qquad s = t/T,\quad \Delta t = 1/T
$$

一句话先说结论：

**离散 DDPM 里的 $\beta_t$，在连续时间视角下，不再是“第 $t$ 步固定加入多少噪声”，而是“单位时间噪声注入速率 $\beta(s)$”乘上“一小步时间长度 $\Delta t$”。**

## 1. 从离散 DDPM 出发

DDPM 单步写作：

$$
x_t = \sqrt{1-\beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon_t,
\qquad \epsilon_t \sim \mathcal N(0, I)
$$

这里的 $\beta_t$ 可以直接理解成：

- 它是第 $t$ 步的噪声量；
- 它决定这一小步里保留多少旧信号、注入多少新噪声。

如果我们只做离散推导，到这里就够了。但如果想把整个过程放到连续时间里看，就要把“第几步”改写成“第几个时间点”。

## 2. 把 $T$ 步放到区间 $[0, 1]$

设整个 forward process 一共有 $T$ 步，并把它均匀铺到时间区间 $[0, 1]$ 上，那么：

$$
\Delta t = \frac{1}{T}
$$

第 $t$ 步对应的连续时间点是：

$$
s = \frac{t}{T} \in [0, 1]
$$

于是，原本的离散序号 $t$，现在就可以看成连续时间变量 $s$ 上的采样点。

## 3. 为什么写成 $\beta_t = \beta(s)\Delta t$

这一步的核心，是把离散量改写成“速率乘步长”的形式。

- 离散视角下，$\beta_t$ 是“这一小步总共加了多少噪声”；
- 连续视角下，$\beta(s)$ 是“单位时间加噪的速率”；
- 那么在长度为 $\Delta t$ 的一小段时间里，实际加入的噪声量就应当近似为：

$$
\beta(s)\Delta t
$$

所以我们写：

$$
\beta_t = \beta\left(\frac{t}{T}\right)\frac{1}{T} = \beta(s)\Delta t
$$

这其实就是一个非常标准的连续化动作：  
**“每一步的总量 = 每单位时间的密度（速率）times 这一步的时间长度。”**

## 4. 一个直觉类比

可以把它类比成“水流”：

- $\beta(s)$：每秒流多少水；
- $\Delta t$：流了多久；
- $\beta(s)\Delta t$：这一小段时间里实际流过了多少水。

同样地，在扩散模型里：

- $\beta(s)$：单位时间噪声强度；
- $\Delta t$：一步的时间长度；
- $\beta_t$：这一步真正注入的噪声量。

所以 $\beta_t$ 不是凭空消失了，而是被重新解释成了一个时间小区间上的累计量。

## 5. 为什么必须这样缩放

连续极限要求：

$$
T \to \infty,\qquad \Delta t \to 0
$$

这时每一步都应该变得非常小。也就是说，单步噪声量 $\beta_t$ 也必须随步长一起缩小，量级应当是：

$$
\beta_t = O(\Delta t)
$$

否则会出问题：

- 如果每一步的 $\beta_t$ 不随 $\Delta t$ 缩小，那么当步数越来越多时，总噪声会爆掉；
- 只有把 $\beta_t$ 写成 $\beta(s)\Delta t$，总噪声积累才会保持有限，并在极限下收敛到一个合法的连续随机过程。

这正是 SDE 极限能成立的关键。

## 6. 代回单步公式，会看到 SDE 的影子

把

$$
\beta_t = \beta(s)\Delta t
$$

代回 DDPM 单步公式：

$$
x_t = \sqrt{1-\beta(s)\Delta t} \cdot x_{t-1} + \sqrt{\beta(s)\Delta t} \cdot \epsilon_t
$$

当 $\Delta t$ 很小时，有一阶近似：

$$
\sqrt{1-\beta(s)\Delta t} \approx 1 - \frac{1}{2}\beta(s)\Delta t
$$

这个近似来自函数

$$
f(x) = \sqrt{1-x}
$$

在 $x=0$ 附近的一阶 Taylor 展开。令

$$
x = \beta(s)\Delta t
$$

则：

$$
f(x) \approx f(0) + f'(0)x
$$

其中

$$
f(0) = 1
$$

并且

$$
f'(x) = -\frac{1}{2}(1-x)^{-1/2}
\quad \Longrightarrow \quad
f'(0) = -\frac{1}{2}
$$

所以

$$
\sqrt{1-x} \approx 1 - \frac{1}{2}x
$$

代回 $x = \beta(s)\Delta t$，就得到

$$
\sqrt{1-\beta(s)\Delta t}
\approx
1 - \frac{1}{2}\beta(s)\Delta t
$$

如果再多看一项，二阶展开是：

$$
\sqrt{1-x}
=
1 - \frac{1}{2}x - \frac{1}{8}x^2 + O(x^3)
$$

所以这里忽略掉的误差量级是

$$
O\big((\beta(s)\Delta t)^2\big)
$$

这也正是为什么当 $\Delta t$ 很小时，一阶近似已经足够好。

于是：

$$
x_t - x_{t-1}
\approx
-\frac{1}{2}\beta(s)x_{t-1}\Delta t
+ \sqrt{\beta(s)\Delta t}\,\epsilon_t
$$

这已经非常接近 Euler-Maruyama 离散化形式了。把

$$
\sqrt{\Delta t}\,\epsilon_t
$$

识别成布朗运动增量 $dW_s$，就得到连续极限：

$$
dx = -\frac{1}{2}\beta(s)x\,ds + \sqrt{\beta(s)}\,dW_s
$$

这就是 DDPM 对应的 VP-SDE。

## 7. 这个式子在表达什么

上面的 SDE 可以拆成两部分：

第一部分：

$$
-\frac{1}{2}\beta(s)x\,ds
$$

表示信号会被逐渐缩小，也就是样本被慢慢往 0 拉。

第二部分：

$$
\sqrt{\beta(s)}\,dW_s
$$

表示系统不断注入高斯噪声，强度由 $\beta(s)$ 控制。

所以它的物理图像是：

$$
\text{信号持续衰减} + \text{噪声持续注入}
$$

离散 DDPM 说的是“每一步加一点噪声”，连续 SDE 说的是“每一瞬间都在加极小的噪声”。

## 8. linear schedule 在连续视角下怎么理解

如果离散模型里使用线性 schedule，意思通常是 $\beta_t$ 随步数 $t$ 逐渐增大。

在连续时间里，对应的就是让 $\beta(s)$ 成为 $s$ 的线性函数，例如：

$$
\beta(s) = \beta_{\min} + (\beta_{\max} - \beta_{\min})s
$$

然后第 $t$ 步真正使用的噪声量是：

$$
\beta_t \approx \beta\left(\frac{t}{T}\right)\frac{1}{T}
$$

所以 schedule 的设计，本质上是在设计一条“连续时间上的加噪速率曲线”。

## 9. 再往前一步：$\bar\alpha(t)$ 的连续表达

离散 DDPM 里有一个关键量：

$$
\bar\alpha_t = \prod_{i=1}^t (1-\beta_i)
$$

它表示经过很多步之后，还剩下多少信号比例。

当每一步都很小时，有近似：

$$
\log(1-\beta_i) \approx -\beta_i
$$

于是：

$$
\log \bar\alpha_t
=
\sum_{i=1}^t \log(1-\beta_i)
\approx
-\sum_{i=1}^t \beta_i
$$

再用

$$
\beta_i \approx \beta(i/T)\Delta t
$$

把求和变成积分，就得到：

$$
\sum_{i=1}^t \beta_i
\approx
\sum_{i=1}^t \beta(i/T)\Delta t
$$

这里令

$$
\tau_i = \frac{i}{T},
\qquad
\Delta t = \frac{1}{T},
\qquad
s = \frac{t}{T}
$$

则上式可写成

$$
\sum_{i=1}^t \beta(\tau_i)\Delta t
$$

这已经是区间 $[0, s]$ 上一个标准的 Riemann 和了。它表示：

- 把时间区间 $[0, s]$ 切成很多小段；
- 每一小段长度都是 $\Delta t$；
- 在每一小段上取一个采样点 $\tau_i$；
- 用 $\beta(\tau_i)\Delta t$ 近似这一小段对总面积的贡献。

当 $T \to \infty$ 时，$\Delta t \to 0$，这些小矩形就会逼近曲线 $\beta(\tau)$ 下的面积，因此

$$
\sum_{i=1}^t \beta(i/T)\Delta t
\to
\int_0^s \beta(\tau)\,d\tau
$$

于是

$$
\log \bar\alpha_t
\approx
-\sum_{i=1}^t \beta_i
\approx
-\sum_{i=1}^t \beta(i/T)\Delta t
\to
-\int_0^s \beta(\tau)\,d\tau
$$

两边取指数，就得到

$$
\bar\alpha(s)
=
\exp\left(
-\int_0^s \beta(\tau)\,d\tau
\right)
$$

如果想看得再严格一点，可以从

$$
\log(1-u) = -u + O(u^2)
$$

出发，其中

$$
u = \beta(i/T)\frac{1}{T}
$$

于是

$$
\log \bar\alpha_t
=
\sum_{i=1}^t \log\left(1-\beta(i/T)\frac{1}{T}\right)
=
-\sum_{i=1}^t \beta(i/T)\frac{1}{T}
+ \sum_{i=1}^t O\left(\frac{1}{T^2}\right)
$$

前一项收敛到积分，后一项总误差至多是

$$
t \cdot O\left(\frac{1}{T^2}\right) = O\left(\frac{1}{T}\right) \to 0
$$

所以连续极限下确实有

$$
\log \bar\alpha(s) = -\int_0^s \beta(\tau)\,d\tau
$$

这说明连续时间里，真正控制“信号还剩多少”的，是累计噪声强度：

$$
\int_0^s \beta(\tau)\,d\tau
$$

## 10. 最核心的三层理解

你可以把整件事记成三层：

1. 离散 DDPM：$\beta_t$ 是第 $t$ 步的噪声量；
2. 连续时间视角：$\beta(s)$ 是单位时间噪声注入速率；
3. 两者关系：$\beta_t = \beta(s)\Delta t$。

所以，“把 $\beta_t$ 写成密度形式”这句话，本质上是在说：

> 把每一步的噪声量，理解成连续时间噪声强度在一个小时间片上的积分近似。

这一步一旦顺了，后面从 DDPM 过渡到 VP-SDE、reverse-time SDE、probability flow ODE，就都会自然很多。

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
