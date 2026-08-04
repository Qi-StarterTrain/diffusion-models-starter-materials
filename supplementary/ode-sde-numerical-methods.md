# ODE 与 SDE 数值方法入门：从 Taylor 展开到 Euler-Maruyama

这份补充阅读用于连接扩散模型中常见的几类采样器：

- ODE sampler：Euler、Heun、Runge-Kutta、DPM-Solver、UniPC
- SDE sampler：Euler-Maruyama、Milstein、Predictor-Corrector

一句话先说结论：

**数值方法的核心思想是：先写出真实解的 Taylor 或 Itô-Taylor 展开，再决定保留哪些项、舍弃哪些项；舍弃项决定局部误差，局部误差在整个积分区间累积后决定全局误差和方法阶数。**

## 1. 为什么需要数值方法

考虑常微分方程 ODE：

$$
\frac{dx}{dt}=f(x,t)
$$

大多数情况下，我们无法写出解析解，所以希望从已知的：

$$
x(t)
$$

近似计算：

$$
x(t+h)
$$

其中：

$$
h=\Delta t
$$

是时间步长。

数值方法要解决的问题就是：

$$
x(t+h)\approx ?
$$

## 2. Taylor 展开是数值方法的起点

注意 $x=x(t)$ 本身就是时间的函数。

因此可以直接对 $x(t+h)$ 做 Taylor 展开：

$$
\boxed{
x(t+h)
=
x(t)
+
h x'(t)
+
\frac{h^2}{2}x''(t)
+
\frac{h^3}{6}x'''(t)
+
\cdots
}
$$

这一步只是普通微积分，还没有用到 ODE。

## 3. 利用 ODE 消去导数

ODE 给出：

$$
x'(t)=f(x,t)
$$

所以 Taylor 展开的第一阶项可以直接写成：

$$
h x'(t)=h f(x,t)
$$

关键是二阶导数 $x''(t)$。

因为：

$$
x'(t)=f(x(t),t)
$$

所以：

$$
x''(t)
=
\frac{d}{dt} f(x(t),t)
$$

这里会出现链式法则。

## 4. 为什么会出现链式法则

因为：

$$
f=f(x,t)
$$

而：

$$
x=x(t)
$$

所以实际上：

$$
f=f(x(t),t)
$$

变量关系可以理解为：

```text
t
│
├──→ x(t)
│      │
│      ▼
└────────→ f(x,t)
```

也就是说，$t$ 一方面直接进入 $f$，另一方面先影响 $x(t)$，再通过 $x$ 影响 $f$。

因此必须使用多元链式法则。

## 5. 多元链式法则

若：

$$
z=f(x,y)
$$

其中：

$$
x=x(t),\qquad y=y(t)
$$

则：

$$
\boxed{
\frac{dz}{dt}
=
\frac{\partial f}{\partial x}\frac{dx}{dt}
+
\frac{\partial f}{\partial y}\frac{dy}{dt}
}
$$

对于 ODE：

$$
f=f(x,t)
$$

第二个变量就是 $t$，所以：

$$
\frac{dt}{dt}=1
$$

于是：

$$
\boxed{
\frac{d}{dt}f(x,t)
=
f_x\frac{dx}{dt}
+
f_t
}
$$

再利用：

$$
\frac{dx}{dt}=f
$$

得到：

$$
\boxed{
x''
=
f_t
+
f_x f
}
$$

这里 $f_x$ 表示 $\partial f/\partial x$，$f_t$ 表示 $\partial f/\partial t$。

## 6. 真实解的 Taylor 展开

把 $x'=f$ 和 $x''=f_t+f_xf$ 代入 Taylor 展开：

$$
\boxed{
x(t+h)
=
x
+
h f
+
\frac12 h^2(f_t+f_xf)
+
O(h^3)
}
$$

如果是自治系统：

$$
f=f(x)
$$

也就是 $f$ 不显式依赖 $t$，那么：

$$
f_t=0
$$

于是：

$$
\boxed{
x(t+h)
=
x
+
h f
+
\frac12 h^2 f'f
+
O(h^3)
}
$$

这是理解 ODE 数值方法最重要的公式之一。

## 7. Euler 方法如何得到

Euler 方法就是只保留 Taylor 展开的第一阶项：

$$
\boxed{
x_{n+1}
=
x_n
+
h f(x_n,t_n)
}
$$

所以 Euler 可以理解成：

> 真实 Taylor 展开截断到一阶。

它没有使用二阶项：

$$
\frac12 h^2(f_t+f_xf)
$$

也没有使用更高阶项。

## 8. 局部截断误差是什么

真实解是：

$$
x
+
h f
+
\frac12 h^2(f_t+f_xf)
+
O(h^3)
$$

Euler 只保留：

$$
x+hf
$$

两者相减，得到一步误差，也叫局部截断误差：

$$
\boxed{
e_{\mathrm{local}}
=
\frac12 h^2(f_t+f_xf)
+
O(h^3)
}
$$

因此：

$$
\boxed{
e_{\mathrm{local}}=O(h^2)
}
$$

这里的 $O(h^2)$ 表示：当 $h\to 0$ 时，误差的主导量级是 $h^2$。

## 9. 为什么可以写成 $O(h^2)$

误差实际上包含很多项：

$$
h^2+h^3+h^4+\cdots
$$

但是当：

$$
h\to 0
$$

时：

$$
h^2 \gg h^3 \gg h^4
$$

因此误差主要由第一个被舍弃项决定。

这个第一个被舍弃项叫：

> Leading Error Term，主导误差项。

所以通常写成：

$$
O(h^2)
$$

## 10. 为什么 Euler 是一阶方法

Euler 的一步局部误差是：

$$
O(h^2)
$$

但是要求解整个区间：

$$
[0,T]
$$

需要：

$$
N=\frac{T}{h}
$$

步。

误差会不断传播并累积。直观估计：

$$
N\cdot O(h^2)
=
\frac{T}{h}O(h^2)
=
O(h)
$$

严格证明需要稳定性分析，例如 Grönwall 不等式，但结论一致：

$$
\boxed{
\text{Global Error}=O(h)
}
$$

因此 Euler 是一阶数值方法。

“一阶”的意思是：

> 步长减半，全局误差大约也减半。

## 11. 更高阶方法如何理解

如果继续恢复 Taylor 的二阶项，可以得到二阶方法。

如果恢复三阶项，可以得到三阶方法。

如果恢复到四阶项，就得到四阶方法，例如经典 RK4。

所以所有高阶 ODE 方法本质上都是：

> 尽可能恢复更多 Taylor 项。

Runge-Kutta 方法的巧妙之处在于，它通常不用显式计算：

$$
x'',\qquad x''',\qquad x^{(4)}
$$

而是通过多次计算：

$$
f(x,t)
$$

来间接恢复这些高阶项。

## 12. SDE 为什么不同

随机微分方程 SDE 写作：

$$
dx=f(x,t)\,dt+g(x,t)\,dW
$$

其中 $W_t$ 是 Brownian motion，也叫 Wiener process。

它有一个关键性质：

$$
\boxed{
\Delta W
=
W_{t+h}-W_t
\sim
\mathcal N(0,h)
}
$$

因此：

$$
\Delta W
=
\sqrt h\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,1)
$$

由于 Brownian motion 几乎处处不可导，普通 Taylor 展开不再适用。

SDE 要使用的是：

> Itô-Taylor 展开。

## 13. Euler-Maruyama 如何得到

对 SDE：

$$
dx=f(x,t)\,dt+g(x,t)\,dW
$$

Itô-Taylor 的一阶截断给出：

$$
\boxed{
x_{n+1}
=
x_n
+
f(x_n,t_n)h
+
g(x_n,t_n)\Delta W_n
}
$$

其中：

$$
\Delta W_n
\sim
\mathcal N(0,h)
$$

也可以写成：

$$
\boxed{
x_{n+1}
=
x_n
+
f(x_n,t_n)h
+
g(x_n,t_n)\sqrt h\,\epsilon_n,
\qquad
\epsilon_n\sim\mathcal N(0,1)
}
$$

这就是 Euler-Maruyama 方法。

它可以理解成：

> Euler 方法在 SDE 上的推广：普通 Euler 的确定性增量之外，多了一个随机增量。

## 14. Euler-Maruyama 为什么不是简单的一阶

SDE 有两种常见误差标准：strong error 和 weak error。

### 14.1 Strong error

strong error 比较的是单条轨迹：

$$
\mathbb E[|X_T-X_T^h|]
$$

其中 $X_T$ 是真实 SDE 解，$X_T^h$ 是步长为 $h$ 的数值解。

Euler-Maruyama 的 strong order 是：

$$
\boxed{
\text{Strong Order}=0.5
}
$$

直觉上，因为 Brownian 增量的尺度是：

$$
\Delta W\sim \sqrt h
$$

所以单条随机轨迹的精确逼近比 ODE 更难。

### 14.2 Weak error

weak error 比较的是分布或期望：

$$
\left|
\mathbb E[\phi(X_T)]
-
\mathbb E[\phi(X_T^h)]
\right|
$$

其中 $\phi$ 是测试函数。

Euler-Maruyama 的 weak order 是：

$$
\boxed{
\text{Weak Order}=1
}
$$

因此有些材料说 Euler-Maruyama 是 first-order，通常指的是 weak first-order，而不是 strong first-order。

## 15. 更高阶 SDE 方法：Milstein

更高阶的 SDE 方法里，最经典的是 Milstein 方法。

一维情形下：

$$
\boxed{
x_{n+1}
=
x_n
+
f h
+
g\Delta W
+
\frac12 gg'
\left((\Delta W)^2-h\right)
}
$$

其中：

$$
g'=\frac{\partial g}{\partial x}
$$

Milstein 方法恢复了 Itô-Taylor 展开中的下一项。

因此它把 strong order 从：

$$
0.5
$$

提高到：

$$
1
$$

在高维或非交换噪声场景下，Milstein 的实现会更复杂，所以 diffusion model 实践中不一定常用它。

## 16. 与 diffusion model 的关系

扩散模型里的反向 SDE 常写成：

$$
dx
=
\left(f-g^2s_\theta\right)dt
+
g\,dW
$$

其中：

$$
s_\theta(x,t)\approx \nabla_x\log p_t(x)
$$

是神经网络估计的 score。

Euler-Maruyama 离散化就是：

$$
x_{k-1}
=
x_k
+
\left(f-g^2s_\theta\right)h
+
g\sqrt h\,\epsilon
$$

其中：

$$
\epsilon\sim\mathcal N(0,I)
$$

这就是很多 reverse SDE sampler 的基本形式。

另一方面，现代 diffusion sampler 更多时候求解的是 probability flow ODE：

$$
\frac{dx}{dt}
=
f
-
\frac12 g^2s_\theta
$$

因此可以使用 ODE 求解器：

- Euler
- Heun
- Runge-Kutta
- DPM-Solver
- UniPC

这些方法的共同目标是：用更少的函数评估次数，更准确地近似从噪声到数据的确定性生成轨迹。

## 17. 一张图串起整个逻辑

ODE 侧：

```text
               ODE
        dx/dt = f(x,t)
                  │
                  ▼
        Taylor 展开 x(t+h)
                  │
                  ▼
      利用链式法则求 x'', x'''
                  │
                  ▼
   x + h f + 1/2 h^2(f_t + f_x f) + ...
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  保留一阶               保留更多项
        │                   │
        ▼                   ▼
    Euler               RK2/RK4/Heun
        │
        ▼
 Local Error = O(h^2)
        │
        ▼
 Global Error = O(h)
        │
        ▼
     一阶方法
```

SDE 侧：

```text
               SDE
        dx = fdt + gdW
                  │
                  ▼
        Itô-Taylor 展开
                  │
                  ▼
      保留 drift 项 + 随机项
                  │
                  ▼
       Euler-Maruyama
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Strong Order = 0.5   Weak Order = 1
                  │
                  ▼
              Milstein
        恢复下一阶 Itô 项
```

## 18. 最后总结

这条知识链可以压缩成三句话：

1. ODE 数值方法从 Taylor 展开出发，Euler 是保留一阶项，RK 类方法是用多次函数评估恢复更高阶项。
2. SDE 数值方法从 Itô-Taylor 展开出发，Euler-Maruyama 是保留 drift 项和 Brownian 随机增量，Milstein 进一步恢复下一阶随机项。
3. Diffusion sampler 的设计，本质上是在选择求解 reverse SDE 还是 probability flow ODE，以及用什么数值方法更快、更稳地近似这条生成过程。
