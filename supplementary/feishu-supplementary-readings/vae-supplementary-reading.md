# VAE 补充阅读材料

**VAE 是一种“带概率建模的自编码器”，它不是直接把图像压缩成一个确定向量，而是把图像编码成一个隐变量分布，然后从这个分布中采样，再解码生成图像。**

---

## 1. VAE 想解决什么问题？
普通 Autoencoder 是：
$$x \rightarrow z \rightarrow \hat{x}$$
也就是：
原始图像 $x$ 经过 Encoder 得到隐变量 $z$，再经过 Decoder 重建出 $\hat{x}$。
但普通 Autoencoder 的问题是：它只会学“重建”，不一定能学到一个适合生成的隐空间。比如你随机采样一个 $z$，Decoder 可能生成很差的图像，因为隐空间不一定是连续、规则的。
VAE 的目标是：

> 学一个规则的隐空间，使得我们可以从中随机采样 $z$，再生成合理的图像 $x$。

---

## 2. VAE 的生成过程
VAE 假设图像是由某个隐变量 (z) 生成的：
$$z \sim p(z), \quad x \sim p_\theta(x|z)$$
其中：
$$p(z)$$
通常是标准高斯分布：
$$p(z)=\mathcal{N}(0,I)$$
也就是说，我们先从一个简单的高斯分布中采样一个隐变量 $z$，然后用 Decoder 生成图像：
$$z \rightarrow x$$
所以 VAE 的生成流程是：
```plaintext
sample z from N(0, I)
        ↓
Decoder pθ(x|z)
        ↓
generate image x
```

---

## 3. 为什么需要 Encoder？
理论上，我们想最大化数据似然：
$$\log p_\theta(x)$$
也就是：模型生成真实图像 $x$ 的概率越大越好。
但是：

$$p_\theta(x)=\int p_\theta(x|z)p(z)dz$$

这个积分通常很难直接算。
所以 VAE 引入一个近似后验分布：
$$q_\phi(z|x)$$
它由 Encoder 预测，用来近似真实后验：
$$p_\theta(z|x)$$
简单说：
```plaintext
Encoder qφ(z|x): 给定图像 x，推断它可能对应哪些 z
Decoder pθ(x|z): 给定隐变量 z，生成图像 x
```

---

## 4. ELBO 是什么？
VAE 的训练目标是最大化 ELBO：
$$\mathcal{L}_{\mathrm{ELBO}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))$$
它包含两项。
第一项：
$$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$$
叫 **重建项**。它希望 Decoder 能够根据 $z$ 尽可能重建原始图像 $x$。
可以理解为：

> 生成出来的图像要像原图。

第二项：
$$D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))$$
叫 **KL 正则项**。它希望 Encoder 预测出来的隐变量分布 $q_\phi(z|x)$ 不要偏离标准高斯先验 $p(z)$ 太远。
可以理解为：

> 每张图像编码出来的隐变量分布，都要尽量贴近 $\mathcal{N}(0,I)$，这样以后随机采样 $z$ 时才能生成合理图像。

---

## 5. ELBO 的直观理解
VAE 同时做两件事：
```plaintext
1. 重建得好：
   z 要包含足够的信息，让 Decoder 能还原 x

2. 隐空间规则：
   z 的分布要接近标准高斯，方便采样和插值
```

所以 VAE 的训练目标可以理解为：
$$
\text{ELBO} = \text{重建质量} - \text{隐空间不规则程度}
$$
也就是：
```plaintext
既要重建好，又要隐空间规整。
```

这就是 VAE 和普通 Autoencoder 最大的区别。

---

## 6. Reparameterization Trick 是关键
VAE 的 Encoder 不直接输出一个确定的 $z$，而是输出一个高斯分布的参数：
$$
q_\phi(z|x)=\mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))
$$
然后从这个分布中采样：
$$
z \sim \mathcal{N}(\mu, \sigma^2)
$$
但采样操作本身不可导，因此不能直接反向传播。
于是 VAE 使用重参数化技巧：
$$
z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
$$
这样随机性被转移到了 $\epsilon$，而 $\mu$ 和 $\sigma$ 仍然可以通过梯度学习。
流程是：
```plaintext
x
↓
Encoder
↓
μ, σ
↓
ε ~ N(0, I)
↓
z = μ + σ · ε
↓
Decoder
↓
x̂
```

---

## 7. 为什么 VAE 训练稳定？
因为 VAE 的训练目标是明确的概率目标：最大化似然下界 ELBO。
相比 GAN，VAE 不需要训练一个 Generator 和 Discriminator 做对抗博弈，因此训练过程更稳定。
GAN 的训练是：
```plaintext
Generator 和 Discriminator 互相博弈
```

VAE 的训练是：
```plaintext
最大化一个明确的优化目标 ELBO
```

所以 VAE 通常更容易训练，不容易出现 GAN 那种 mode collapse 或训练震荡。

---

## 8. 为什么 VAE 生成图像偏模糊？
这是 VAE 的经典问题。
很多 VAE 使用高斯解码器：
$$
p_\theta(x|z)=\mathcal{N}(\mu_\theta(z), \sigma^2 I)
$$
在这种假设下，最大化重建概率等价于最小化 MSE。
而 MSE 的问题是：它倾向于生成“平均结果”。
举个例子，如果一张图像中同一个位置可能是眼睛、头发、背景等多种可能，MSE 会倾向于把这些可能性平均起来，于是图像就变模糊了。
可以直观理解为：
```plaintext
真实图像可能有多个清晰解
MSE 倾向于取平均
平均之后就变模糊
```

这就是为什么早期 VAE 生成的图像通常比 GAN 模糊。

---

## 9. 什么是 posterior collapse？
Posterior collapse，即后验坍塌，指的是：
$$
q_\phi(z|x) \approx p(z)
$$
也就是说，Encoder 输出的后验分布几乎退化成先验分布，隐变量 $z$ 不再包含输入 $x$ 的有效信息。
此时 Decoder 几乎不依赖 $z$，而是自己学会生成数据。
这在强 Decoder 中尤其常见，比如文本生成模型或自回归 Decoder。
直观解释：
```plaintext
Encoder：我给你 z，里面包含 x 的信息
Decoder：不用了，我自己能生成
于是 z 被忽略
```

结果就是隐变量失去作用。

---

## 10. VAE 的优点总结
VAE 的优点主要是：
```plaintext
1. 训练稳定
2. 有明确的概率目标
3. 可以估计 likelihood 下界
4. 隐空间连续、规整
5. 适合插值、编辑、表示学习
```

例如在隐空间中做插值：
$$
z_1 \rightarrow z_2
$$
通常会得到比较平滑的语义变化。
这说明 VAE 学到的隐空间具有一定结构。

---

## 11. VAE 的缺点总结
VAE 的主要缺点是：
```plaintext
1. 图像容易模糊
2. 高斯解码器表达能力有限
3. KL 项可能压制隐变量表达能力
4. 强 Decoder 下容易 posterior collapse
5. 原始 VAE 的样本质量通常不如 GAN / Diffusion
```

---

## 一句话总结
**VAE 通过 Encoder 学习 (q_\phi(z|x))，通过 Decoder 学习 (p_\theta(x|z))，并用 ELBO 同时约束“重建质量”和“隐空间规则性”，从而得到一个可以采样、插值和编辑的生成模型。**
可以把 VAE 理解成：
```plaintext
Autoencoder + 概率建模 + 隐空间正则化
```

它的核心价值不只是生成图像，而是学习一个结构良好的 latent space。
