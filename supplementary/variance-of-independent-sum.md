# 独立随机变量相加方差推导

这个公式来自**方差的定义**。
证明：

```math
\mathrm{Var}(aA+bB)=a^2\mathrm{Var}(A)+b^2\mathrm{Var}(B)
```

但严格来说，它成立需要一个条件：

```math
A \text{ 和 } B \text{ 相互独立，或至少不相关}
```

更一般的公式其实是：

```math
\mathrm{Var}(aA+bB)=a^2\mathrm{Var}(A)+b^2\mathrm{Var}(B)+2ab\mathrm{Cov}(A,B)
```

当 (A) 和 (B) 独立时：

```math
\mathrm{Cov}(A,B)=0
```

所以就变成：

```math
\mathrm{Var}(aA+bB)=a^2\mathrm{Var}(A)+b^2\mathrm{Var}(B)
```

下面详细推导。

## 1. 方差的定义

随机变量 (X) 的方差定义为：

```math
\mathrm{Var}(X)=\mathbb E\left[(X-\mathbb E[X])^2\right]
```

也可以写成：

```math
\mathrm{Var}(X)=\mathbb E[X^2]-(\mathbb E[X])^2
```

现在令：

```math
X=aA+bB
```

我们要求：

```math
\mathrm{Var}(aA+bB)
```

---

## 2. 先计算均值

根据期望的线性性：

```math
\mathbb E[aA+bB]=a\mathbb E[A]+b\mathbb E[B]
```

所以：

```math
X-\mathbb E[X]=
aA+bB-\left(a\mathbb E[A]+b\mathbb E[B]\right)
```

整理得：

```math
X-\mathbb E[X]=
a(A-\mathbb E[A])+b(B-\mathbb E[B])
```

---

## 3. 代入方差定义

```math
\mathrm{Var}(aA+bB)=
\mathbb E\left[
\left(
a(A-\mathbb E[A])+b(B-\mathbb E[B])
\right)^2
\right]
```

把平方展开：

```math
\left(
a(A-\mathbb E[A])+b(B-\mathbb E[B])
\right)^2
```

```math
a^2(A-\mathbb E[A])^2
+
b^2(B-\mathbb E[B])^2
+
2ab(A-\mathbb E[A])(B-\mathbb E[B])
```

因此：

```math
\mathrm{Var}(aA+bB)=
\mathbb E\left[
a^2(A-\mathbb E[A])^2
+
b^2(B-\mathbb E[B])^2
+
2ab(A-\mathbb E[A])(B-\mathbb E[B])
\right]
```

利用期望的线性性：

```math
\mathrm{Var}(aA+bB)=
a^2\mathbb E[(A-\mathbb E[A])^2]
+
b^2\mathbb E[(B-\mathbb E[B])^2]
+
2ab\mathbb E[(A-\mathbb E[A])(B-\mathbb E[B])]
```

---

## 4. 识别方差和协方差

根据定义：

```math
\mathbb E[(A-\mathbb E[A])^2]=\mathrm{Var}(A)
```

```math
\mathbb E[(B-\mathbb E[B])^2]=\mathrm{Var}(B)
```

而：

```math
\mathbb E[(A-\mathbb E[A])(B-\mathbb E[B])]
```

就是 (A) 和 (B) 的协方差：

```math
\mathrm{Cov}(A,B)
```

所以得到：

```math
\mathrm{Var}(aA+bB)=
a^2\mathrm{Var}(A)
+
b^2\mathrm{Var}(B)
+
2ab\mathrm{Cov}(A,B)
```

这就是完整公式。

## 5. 为什么 DDPM 里可以去掉协方差项？

在 DDPM forward process 中：

```math
x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t
```

这里：

```math
A=x_{t-1}
```

```math
B=\epsilon_t
```

新噪声 (\epsilon_t) 是每一步独立采样的高斯噪声，因此它和 (x_{t-1}) 独立。
所以：

```math
\mathrm{Cov}(x_{t-1},\epsilon_t)=0
```

因此：

```math
\mathrm{Var}(x_t)=
(1-\beta_t)\mathrm{Var}(x_{t-1})
+
\beta_t\mathrm{Var}(\epsilon_t)
```

这就是前面使用的公式来源。

## 6. 一个更直观的理解

如果两个随机变量独立，那么它们的波动不会互相“配合”。
例如：

```math
aA
```

带来的波动大小是：

```math
a^2\mathrm{Var}(A)
```

因为方差对缩放系数是平方关系：

```math
\mathrm{Var}(aA)=a^2\mathrm{Var}(A)
```

同理：

```math
\mathrm{Var}(bB)=b^2\mathrm{Var}(B)
```

如果 (A) 和 (B) 独立，它们之间没有额外相关性，所以总方差就是两部分方差直接相加：

```math
\mathrm{Var}(aA+bB)=
\mathrm{Var}(aA)+\mathrm{Var}(bB)
```

也就是：

```math
a^2\mathrm{Var}(A)+b^2\mathrm{Var}(B)
```

---

所以总结一下：

```math
\mathrm{Var}(aA+bB)=a^2\mathrm{Var}(A)+b^2\mathrm{Var}(B)
```

不是凭空来的，而是从方差定义展开得到的；它省略了协方差项，前提是：

```math
A,B \text{ 独立或不相关}
```

在 DDPM 中，这个条件成立，因为新加入的噪声 (\epsilon_t) 是独立采样的。
