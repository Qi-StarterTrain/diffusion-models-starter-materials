# Probability Flow ODE：如何理解“边缘分布完全相同”

这份补充阅读对应 `derive_04_score_sde.md` 的 §4，专门解释 probability flow ODE 中这句话：

> 它在每个时间 $t$ 的边缘分布 $p_t(x)$ 与原 SDE 完全相同，但轨迹是确定性的。

一句话先说结论：

**“边缘分布完全相同”指的是：在任意固定时间 $t$，如果只看所有样本点 $x_t$ 的整体分布，SDE 和 probability flow ODE 得到的是同一个 $p_t(x)$；但如果追踪某个单独样本从 $0$ 到 $T$ 的整条路径，两者通常不同。**

## 1. SDE 看的是随机轨迹

给定一个 forward SDE：

$$
dx = f(x,t)\,dt + g(t)\,dW
$$

从同一个初始分布采样：

$$
x_0 \sim p_0(x)
$$

然后让很多样本点沿这个 SDE 演化。因为有 $dW$，每个样本都会受到随机扰动，所以单个样本轨迹是随机的。

也就是说，即使两个样本从同一个起点出发，只要布朗运动噪声不同，它们之后的路径也会不同。

## 2. ODE 看的是确定性轨迹

probability flow ODE 写作：

$$
\frac{dx}{dt}
=
f(x,t)
-
\frac{1}{2}g(t)^2\nabla_x\log p_t(x)
$$

这里没有 $dW$，所以给定初始点 $x_0$ 后，轨迹就是确定的。

如果同一个 $x_0$ 反复运行这个 ODE，只要数值求解器和步长相同，得到的路径就相同。

## 3. “边缘分布相同”是什么意思

考虑整个随机过程：

$$
(x_0, x_{t_1}, x_{t_2}, \dots, x_T)
$$

这描述的是一整条路径的联合分布。

而“边缘分布”只关心某一个时间点，例如：

$$
p_t(x)
$$

它回答的是：

> 在时间 $t$，样本点落在位置 $x$ 附近的概率密度是多少？

所以 probability flow ODE 与原 SDE 的关系是：

$$
x_t^{\mathrm{SDE}} \sim p_t(x)
$$

并且：

$$
x_t^{\mathrm{ODE}} \sim p_t(x)
$$

对每个固定的 $t$ 都成立。

但是，这不代表：

$$
(x_0^{\mathrm{SDE}}, x_t^{\mathrm{SDE}}, x_T^{\mathrm{SDE}})
$$

和

$$
(x_0^{\mathrm{ODE}}, x_t^{\mathrm{ODE}}, x_T^{\mathrm{ODE}})
$$

这两条路径的联合分布相同。

更短地说：

- **边缘分布相同**：每个时间点的“样本云形状”一样；
- **路径分布不同**：样本从哪里来、怎么走过去，通常不一样。

## 4. 一个直观比喻

想象你在多个时间点给一群粒子拍照。

SDE 中，粒子一边被 drift 推动，一边受到随机噪声扰动。

ODE 中，粒子没有随机噪声，而是沿一个确定性速度场移动。

probability flow ODE 的“神奇”之处是：

**虽然每个粒子的运动路线不同，但在任意时间 $t$ 拍一张全体粒子的分布照片，ODE 和 SDE 的照片完全一样。**

## 5. 为什么确定性 ODE 能模拟随机 SDE 的密度变化

核心原因是：SDE 的 Fokker-Planck 方程可以改写成一个连续性方程。

原 SDE：

$$
dx = f(x,t)\,dt + g(t)\,dW
$$

对应 Fokker-Planck 方程：

$$
\frac{\partial p_t}{\partial t}
=
-\nabla_x\cdot(f p_t)
+
\frac{1}{2}g(t)^2\nabla_x^2 p_t
$$

其中第一项是 drift 对概率密度的搬运，第二项是噪声对概率密度的扩散。

利用恒等式：

$$
\nabla_x^2 p_t
=
\nabla_x\cdot(\nabla_x p_t)
$$

以及：

$$
\nabla_x p_t
=
p_t \nabla_x \log p_t
$$

所以：

$$
\nabla_x^2 p_t
=
\nabla_x\cdot(p_t\nabla_x\log p_t)
$$

代回 Fokker-Planck 方程：

$$
\frac{\partial p_t}{\partial t}
=
-\nabla_x\cdot(f p_t)
+
\nabla_x\cdot\left(
\frac{1}{2}g(t)^2 p_t\nabla_x\log p_t
\right)
$$

合并成：

$$
\frac{\partial p_t}{\partial t}
=
-\nabla_x\cdot
\left[
\left(
f
-
\frac{1}{2}g(t)^2\nabla_x\log p_t
\right)p_t
\right]
$$

这正好是一个 ODE：

$$
\frac{dx}{dt}=h(x,t)
$$

对应的密度演化方程：

$$
\frac{\partial p_t}{\partial t}
=
-\nabla_x\cdot(hp_t)
$$

因此只要令：

$$
h(x,t)
=
f(x,t)
-
\frac{1}{2}g(t)^2\nabla_x\log p_t(x)
$$

就得到：

$$
\boxed{
\frac{dx}{dt}
=
f(x,t)
-
\frac{1}{2}g(t)^2\nabla_x\log p_t(x)
}
$$

这就是 probability flow ODE。

## 6. score 项在做什么

score 是：

$$
\nabla_x \log p_t(x)
$$

它指向当前分布密度上升最快的方向，也就是高密度区域。

因此：

$$
-\nabla_x \log p_t(x)
$$

会指向低密度方向。

在 probability flow ODE 中：

$$
-\frac{1}{2}g(t)^2\nabla_x\log p_t(x)
$$

这项可以理解为：用一个确定性速度场，把样本从高密度区域向外推，从而制造出和随机噪声相同的密度扩散效果。

也就是说：

**SDE 用随机噪声让样本云变宽；probability flow ODE 用 score 修正后的确定性速度场让样本云以同样方式变宽。**

## 7. 对扩散模型的意义

扩散模型本质上关心的是分布演化：

$$
p_0(x)
\rightarrow
p_t(x)
\rightarrow
p_T(x)
$$

SDE 给出一种随机采样路径，ODE 给出一种确定性采样路径。

由于两者在每个时间点的边缘分布相同，所以从 $p_T$ 反向积分 probability flow ODE，也可以回到 $p_0$。

这就是 DDIM、DPM-Solver、EDM 等快速采样器背后的核心思想：

> 把随机扩散采样改写成确定性 ODE 求解问题。

## 8. 为什么这个 ODE 重要

一旦把随机的反向 SDE 改写成确定性的 probability flow ODE，扩散模型采样就从“模拟随机去噪过程”变成了“求解常微分方程”。

这会带来几个重要好处。

### 8.1 可逆

ODE 是确定性动力系统：

$$
\frac{dx}{dt}=h(x,t)
$$

给定某个初始点 $x_T$，它会对应唯一一条轨迹。

所以理论上，如果先从 $x_T$ 倒着积分到 $x_0$：

$$
x_T \rightarrow x_0
$$

再沿同一个 ODE 正着积分回去：

$$
x_0 \rightarrow x_T
$$

就应该回到同一个点。

SDE 则不同。SDE 每一步都有随机噪声：

$$
dx=fdt+gdW
$$

即使从同一个 $x_T$ 出发，只要重新采样噪声，路径就会变化。因此 SDE 不能简单地按同一条轨迹“倒回去”。

实际数值计算中，由于步长有限、score 网络有误差，ODE 的可逆性通常是近似的，而不是完美精确的。

### 8.2 可计算 likelihood

生成模型有时不只想采样，还想知道某个样本的概率密度：

$$
\log p(x)
$$

ODE 有一个很重要的公式，叫 instantaneous change of variables：

$$
\frac{d\log p_t(x_t)}{dt}
=
-\nabla_x\cdot h(x_t,t)
$$

其中：

$$
h(x,t)
=
f(x,t)
-
\frac{1}{2}g(t)^2\nabla_x\log p_t(x)
$$

这个公式的意思是：沿着 ODE 轨迹走时，样本密度的变化率由速度场的散度决定。

因此，如果把数据点 $x_0$ 正向积分到简单分布 $x_T$，就可以用：

$$
\log p_0(x_0)
=
\log p_T(x_T)
+
\int_0^T \nabla_x\cdot h(x_t,t)\,dt
$$

来计算它的 likelihood。这里 $p_T$ 通常是标准高斯或近似标准高斯，所以 $\log p_T(x_T)$ 可以直接计算。

这让 diffusion model 在 ODE 视角下拥有类似 normalizing flow 的密度计算能力。

在高维图像模型中，散度项 $\nabla_x\cdot h$ 通常不会直接完整计算，而是用 Hutchinson trace estimator 等方法估计。

### 8.3 可用高阶 ODE 求解器加速采样

如果用 SDE 采样，常见离散化是 Euler-Maruyama：

$$
x_{t-\Delta t}
=
x_t
+
\text{drift}\cdot\Delta t
+
\text{noise}\cdot\sqrt{\Delta t}
$$

随机项会引入额外方差，通常需要很多步才能稳定。

但 probability flow ODE 没有随机项：

$$
\frac{dx}{dt}=h(x,t)
$$

所以可以使用成熟的 ODE 数值求解器，例如：

- Euler
- Runge-Kutta
- Heun
- DPM-Solver

高阶求解器能更准确地近似轨迹弯曲，因此可以显著减少采样步数。

原始 DDPM 采样常用约 $1000$ 步，而 DDIM、DPM-Solver 等方法可以把步数降到 $50$ 步、$20$ 步甚至更少。

### 8.4 去随机性

probability flow ODE 没有随机噪声项。

所以只要初始噪声 $x_T$ 固定，生成结果就是固定的：

$$
x_T \mapsto x_0
$$

这对调试、复现实验、图像编辑很有用。

多样性并没有消失，因为仍然可以通过换不同的初始噪声 $x_T$ 得到不同样本。去随机性只表示：

$$
\text{同一个初始噪声} \rightarrow \text{同一个输出}
$$

### 8.5 为什么说这是 DDIM 的数学基础

DDPM 的 ancestral sampling 是随机的，形式上更像反向 SDE 的离散化。

DDIM 引入了一个确定性采样版本。当 DDIM 中随机噪声参数设为：

$$
\eta = 0
$$

采样过程就没有额外随机噪声，变成确定性更新：

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
$$

这和 probability flow ODE 的思想一致：

> 用确定性轨迹连接噪声分布和数据分布。

更准确地说：

$$
\text{DDIM}
\approx
\text{probability flow ODE 的一阶离散化}
$$

这里的“一阶离散化”可以理解为：不用连续求解 ODE，而是用有限步长一步一步近似它，类似 Euler 法。

## 9. 最重要的区分

| 角度 | SDE | Probability Flow ODE |
|---|---|---|
| 单个轨迹 | 随机 | 确定 |
| 是否有噪声项 | 有 $dW$ | 没有 |
| 每个时间点的边缘分布 | $p_t(x)$ | 同一个 $p_t(x)$ |
| 路径分布 | 通常不同 | 通常不同 |
| 采样方式 | 随机采样 | 确定性积分 |
| 工程意义 | PC sampler、reverse SDE | DDIM、DPM-Solver、高阶 ODE solver |

最后再压缩成一句：

**probability flow ODE 不是在复现 SDE 中每个样本的随机路径，而是在复现 SDE 对整体概率密度 $p_t(x)$ 的演化。**

另一个角度看：

**probability flow ODE 重要，是因为它把扩散模型从“模拟随机去噪过程”变成了“求解确定性生成轨迹”，从而带来可逆、可算 likelihood、可加速和可复现。**
