# DDPM 方差稳定性推导

在 DDPM 的 forward diffusion 中，标准定义是：

$$
q(x_t\mid x_{t-1})=\mathcal N\left(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I\right)
$$

也就是说：

$$
x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t,\qquad \epsilon_t\sim \mathcal N(0,I)
$$

而不是简单写成：

$$
x_t=x_{t-1}+\sqrt{\beta_t}\epsilon_t
$$

核心原因是：**DDPM 希望每一步加入噪声，但同时保持整体尺度稳定，不让方差无限增长。**

## 1. 标准 DDPM 加噪公式

设：

$$
x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t
$$

其中：

$$
\epsilon_t\sim \mathcal N(0,I)
$$

并且假设：

$$
x_{t-1}\ \text{和}\ \epsilon_t\ \text{相互独立}
$$

现在计算 (x_t) 的方差。

## 2. 方差计算公式

如果两个随机变量 (A,B) 相互独立，则：

$$
\mathrm{Var}(aA+bB)=a^2\mathrm{Var}(A)+b^2\mathrm{Var}(B)
$$

这里令：

$$
A=x_{t-1},\qquad B=\epsilon_t
$$

$$
a=\sqrt{1-\beta_t},\qquad b=\sqrt{\beta_t}
$$

所以：

$$
\mathrm{Var}(x_t)=
\mathrm{Var}\left(\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t\right)
$$

展开得到：

$$
\mathrm{Var}(x_t)=
\left(\sqrt{1-\beta_t}\right)^2\mathrm{Var}(x_{t-1})
+
\left(\sqrt{\beta_t}\right)^2\mathrm{Var}(\epsilon_t)
$$

因此：

$$
\mathrm{Var}(x_t)=
(1-\beta_t)\mathrm{Var}(x_{t-1})
+
\beta_t\mathrm{Var}(\epsilon_t)
$$

由于通常设：

$$
\mathrm{Var}(x_{t-1})=1,\qquad \mathrm{Var}(\epsilon_t)=1
$$

所以：

$$
\mathrm{Var}(x_t)=
(1-\beta_t)\cdot 1+\beta_t\cdot 1
$$

$$
\mathrm{Var}(x_t)=1
$$

这说明：**如果上一步的方差是 1，那么下一步的方差仍然是 1。**

## 3. 为什么这个系数是 $\sqrt{1-\beta_t}$

注意，方差会受到系数平方的影响。
如果希望 $x_{t-1}$ 那部分贡献的方差是：

$$
1-\beta_t
$$

那么它前面的系数必须是：

$$
\sqrt{1-\beta_t}
$$

因为：

$$
\mathrm{Var}\left(\sqrt{1-\beta_t}x_{t-1}\right)=
(1-\beta_t)\mathrm{Var}(x_{t-1})
$$

同理，如果希望噪声项贡献的方差是：

$$
\beta_t
$$

那么噪声前面的系数必须是：

$$
\sqrt{\beta_t}
$$

因为：

$$
\mathrm{Var}\left(\sqrt{\beta_t}\epsilon_t\right)=
\beta_t\mathrm{Var}(\epsilon_t)
$$

所以标准形式本质上是在做一个**方差配平**：

$$
\underbrace{1-\beta_t}_{\text{保留原信号的方差比例}}
+
\underbrace{\beta_t}_{\text{新加入噪声的方差比例}}
=1
$$

---

## 4. 如果用简单加法会怎样？

假设错误地定义为：

$$
x_t=x_{t-1}+\sqrt{\beta_t}\epsilon_t
$$

那么：

$$
\mathrm{Var}(x_t)=
\mathrm{Var}(x_{t-1})
+
\beta_t\mathrm{Var}(\epsilon_t)
$$

如果：

$$
\mathrm{Var}(x_{t-1})=1,\qquad \mathrm{Var}(\epsilon_t)=1
$$

则：

$$
\mathrm{Var}(x_t)=1+\beta_t
$$

下一步继续：

$$
x_{t+1}=x_t+\sqrt{\beta_{t+1}}\epsilon_{t+1}
$$

则：

$$
\mathrm{Var}(x_{t+1})=
\mathrm{Var}(x_t)+\beta_{t+1}
$$

代入：

$$
\mathrm{Var}(x_{t+1})=
1+\beta_t+\beta_{t+1}
$$

递推下去：

$$
\mathrm{Var}(x_t)=
1+\sum_{s=1}^t\beta_s
$$

这说明方差会随着时间步不断累积。
如果扩散步数很多，比如 (T=1000)，那么即使每个 $\beta_t$ 很小，累积之后也会显著改变数据尺度。

## 5. 标准 DDPM 的递推更稳定

标准形式为：

$$
x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t
$$

记：

$$
\alpha_t=1-\beta_t
$$

则：

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t
$$

如果 (\mathrm{Var}(x_{t-1})=1)，则：

$$
\mathrm{Var}(x_t)=
\alpha_t\cdot 1+(1-\alpha_t)\cdot 1
$$

$$
\mathrm{Var}(x_t)=1
$$

因此每一步都是：

$$
\text{减少一部分旧信号}+\text{加入等量方差的新噪声}
$$

不是简单地“往原图上堆噪声”，而是：

$$
\text{把一部分信号替换成噪声}
$$

这点非常关键。

## 6. 从多步角度看

DDPM 还有一个重要闭式形式：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
$$

其中：

$$
\bar{\alpha}_t=\prod_{s=1}^t\alpha_s
$$

也就是：

$$
\bar{\alpha}_t=\prod_{s=1}^t(1-\beta_s)
$$

这个形式说明，经过 (t) 步之后：

$$
x_t
$$

可以看成是原始数据 $x_0$ 和标准高斯噪声 $\epsilon$ 的加权组合。
如果 $\mathrm{Var}(x_0)=1$，$\mathrm{Var}(\epsilon)=1$，则：

$$
\mathrm{Var}(x_t)=
\bar{\alpha}_t\cdot 1+(1-\bar{\alpha}_t)\cdot 1
$$

$$
\mathrm{Var}(x_t)=1
$$

所以无论扩散多少步，只要这个形式成立，整体方差都保持稳定。

## 7. 直观理解

$$
x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t
$$

可以理解为：

$$
x_t = \text{保留 } 1-\beta_t \text{ 比例的旧信息} + \text{注入 } \beta_t \text{ 比例的新噪声}
$$

注意这里的“比例”是**方差比例**，不是数值系数比例。
因此数值系数要取平方根：

$$
\sqrt{1-\beta_t},\qquad \sqrt{\beta_t}
$$

而不是：

$$
1-\beta_t,\qquad \beta_t
$$

---

总结一句话：
**DDPM 不是简单地不断加噪声，而是在每一步用噪声替换掉一小部分原有信号；$\sqrt{1-\beta_t}$ 的作用就是缩小旧信号，使旧信号方差和新噪声方差之和仍然为 1，从而保证扩散过程的数值尺度稳定。**
