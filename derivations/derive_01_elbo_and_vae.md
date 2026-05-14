# Derivation 01: ELBO 与 VAE 完整推导

> 本手稿对应 L02 内容，给出 VAE 中 ELBO 的完整、严格推导。
> 建议配合 L02 一起阅读。

---

## §1 设置

我们有数据 $\{x_i\}_{i=1}^N$，希望学习生成模型 $p_\theta(x)$。

引入隐变量 $z$：
$$p_\theta(x) = \int p_\theta(x, z) \, dz = \int p_\theta(x | z) \cdot p(z) \, dz$$

其中：
- $p(z) = \mathcal{N}(0, I)$ 是先验，固定
- $p_\theta(x | z)$ 由解码器神经网络参数化

---

## §2 直接最大化对数似然的困难

$$\log p_\theta(x) = \log \int p_\theta(x | z) \, p(z) \, dz$$

**问题**：
1. 积分一般无解析解
2. 蒙特卡洛估计 $\frac{1}{K} \sum_k p_\theta(x | z_k)$（$z_k \sim p(z)$）在高维下采样效率极低——大多数 $z_k$ 给出 $p_\theta(x | z_k) \approx 0$

**解决思路**：引入"好的"提议分布 $q_\phi(z | x)$，让 $z$ 采样集中在有意义的区域。

---

## §3 ELBO 推导（路径 A：Jensen 不等式）

引入任意分布 $q_\phi(z | x)$：

$$
\begin{aligned}
\log p_\theta(x)
&= \log \int p_\theta(x, z) \, dz \\
&= \log \int q_\phi(z|x) \cdot \frac{p_\theta(x, z)}{q_\phi(z|x)} \, dz \\
&= \log \mathbb{E}_{z \sim q_\phi(z|x)} \left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right] \\
&\geq \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right] \quad \text{(Jensen, } \log \text{ 是凹函数)} \\
&=: \mathrm{ELBO}(x; \theta, \phi)
\end{aligned}
$$

**Jensen 不等式**：对凹函数 $\phi$，$\phi(\mathbb{E}[X]) \geq \mathbb{E}[\phi(X)]$。这里 $\phi = \log$。

---

## §4 ELBO 的两种等价形式

### 4.1 联合分布形式（path A 直接结果）

$$\mathrm{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x, z) - \log q_\phi(z|x)]$$

---

### 4.2 重构 + KL 正则形式（最常用）

把 $p_\theta(x, z) = p_\theta(x | z) \cdot p(z)$ 代入：

$$
\begin{aligned}
\mathrm{ELBO}
&= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x | z) + \log p(z) - \log q_\phi(z|x)] \\
&= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x | z)] + \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(z)}{q_\phi(z|x)}\right] \\
&= \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x | z)]}_{\text{重构项}} - \underbrace{D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))}_{\text{正则项}}
\end{aligned}
$$

最终形式：

$$\boxed{\mathrm{ELBO}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x | z)] - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))}$$

---

## §5 ELBO 与真实对数似然的差距

### 5.1 推导（路径 B：直接展开）

$$
\begin{aligned}
\log p_\theta(x)
&= \log p_\theta(x) \cdot \int q_\phi(z|x) \, dz \quad \text{(乘以 1)} \\
&= \int q_\phi(z|x) \log p_\theta(x) \, dz \\
&= \int q_\phi(z|x) \log \frac{p_\theta(x, z)}{p_\theta(z|x)} \, dz \\
&= \int q_\phi(z|x) \log \frac{p_\theta(x, z) \cdot q_\phi(z|x)}{p_\theta(z|x) \cdot q_\phi(z|x)} \, dz \\
&= \underbrace{\int q_\phi(z|x) \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \, dz}_{\mathrm{ELBO}}
\quad + \quad
\underbrace{\int q_\phi(z|x) \log \frac{q_\phi(z|x)}{p_\theta(z|x)} \, dz}_{D_{\mathrm{KL}}(q_\phi \| p_\theta^{\text{post}})}
\end{aligned}
$$

最终：

$$\boxed{\log p_\theta(x) = \mathrm{ELBO}(x; \theta, \phi) + D_{\mathrm{KL}}(q_\phi(z|x) \| p_\theta(z|x))}$$

由于 KL 非负，自动得到 $\log p_\theta(x) \geq \mathrm{ELBO}$。

---

### 5.2 解读

- **差距 = $D_{\mathrm{KL}}(q_\phi(z|x) \| p_\theta(z|x))$**：变分后验 $q_\phi$ 与真实后验的 KL
- 当 $q_\phi = p_\theta^{\text{post}}$ 时，ELBO = 真实对数似然
- 最大化 ELBO **同时**做两件事：
  1. 提高真实 $\log p_\theta(x)$
  2. 让 $q_\phi$ 逼近真实后验

---

## §6 VAE 的具体实现

### 6.1 设计选择

VAE 假设：
- $p(z) = \mathcal{N}(0, I)$
- $q_\phi(z | x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) \cdot I)$（对角协方差）
- $p_\theta(x | z) = \mathcal{N}(\mu_\theta(z), I)$（单位方差，简化）

---

### 6.2 重构项化简

由于 $p_\theta(x | z) = \mathcal{N}(\mu_\theta(z), I)$：

$$\log p_\theta(x | z) = -\frac{1}{2} \| x - \mu_\theta(z) \|^2 + \text{const}$$

所以：
$$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] = -\frac{1}{2} \mathbb{E}_{q_\phi}[\| x - \mu_\theta(z) \|^2] + \text{const}$$

—— **MSE 重构损失**！

---

### 6.3 KL 项化简（高斯之间）

对 $q_\phi(z|x) = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$ 和 $p(z) = \mathcal{N}(0, I)$：

**一维形式**：
$$D_{\mathrm{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \frac{1}{2}(\mu^2 + \sigma^2 - \log \sigma^2 - 1)$$

**完整推导**：

$$
\begin{aligned}
D_{\mathrm{KL}}
&= \int q(z) \log \frac{q(z)}{p(z)} dz \\
&= \int \mathcal{N}(z; \mu, \sigma^2) \cdot \left[\log \mathcal{N}(z; \mu, \sigma^2) - \log \mathcal{N}(z; 0, 1)\right] dz \\
&= \int q(z) \left[-\frac{(z-\mu)^2}{2\sigma^2} - \log\sigma - \frac{1}{2}\log(2\pi) + \frac{z^2}{2} + \frac{1}{2}\log(2\pi)\right] dz \\
&= \int q(z) \left[-\frac{(z-\mu)^2}{2\sigma^2} - \log\sigma + \frac{z^2}{2}\right] dz \\
&= -\frac{1}{2} \cdot \underbrace{\frac{\mathbb{E}[(z-\mu)^2]}{\sigma^2}}_{=1} - \log\sigma + \frac{1}{2}\mathbb{E}[z^2] \\
&= -\frac{1}{2} - \log\sigma + \frac{1}{2}(\mu^2 + \sigma^2)
\end{aligned}
$$

整理：
$$D_{\mathrm{KL}} = \frac{1}{2}(\mu^2 + \sigma^2 - 2\log\sigma - 1) = \frac{1}{2}(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$$

**$d$ 维形式**（独立各维）：
$$D_{\mathrm{KL}}(\mathcal{N}(\mu, \mathrm{diag}(\sigma^2)) \| \mathcal{N}(0, I)) = \frac{1}{2} \sum_{i=1}^d (\mu_i^2 + \sigma_i^2 - \log\sigma_i^2 - 1)$$

---

### 6.4 VAE 最终训练目标

把两项合起来：

$$\boxed{\mathcal{L}_{\mathrm{VAE}}(\theta, \phi; x) = \underbrace{\| x - \mu_\theta(z) \|^2}_{\text{recon (MSE)}} + \underbrace{\sum_i (\mu_i^2 + \sigma_i^2 - \log\sigma_i^2 - 1)}_{\text{KL，闭合}}}$$

去掉了无关常数（$\frac{1}{2}$ 等可吸收到 learning rate）。

---

## §7 重参数化技巧

**问题**：上述 loss 的重构项含 $\mathbb{E}_{q_\phi(z|x)}[\cdot]$，要对 $\phi$ 求导。

直接 $z \sim q_\phi(z|x)$ 是采样操作，不可导（采样过程梯度不能传过去）。

**重参数化**：
$$z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

把随机性"外包"给 $\epsilon$，$\mu, \sigma$ 与 $z$ 是确定性映射。

**结果**：梯度可以经由 $z$ 流回 $\phi$。

实践代码：
```python
def encode(self, x):
    mu = self.fc_mu(self.encoder(x))
    log_sigma = self.fc_log_sigma(self.encoder(x))  # 注意：预测 log σ
    return mu, log_sigma

def reparameterize(self, mu, log_sigma):
    sigma = torch.exp(log_sigma)
    epsilon = torch.randn_like(sigma)
    return mu + sigma * epsilon

def forward(self, x):
    mu, log_sigma = self.encode(x)
    z = self.reparameterize(mu, log_sigma)
    x_recon = self.decoder(z)
    return x_recon, mu, log_sigma

def vae_loss(x, x_recon, mu, log_sigma):
    sigma2 = torch.exp(2 * log_sigma)
    recon = F.mse_loss(x_recon, x, reduction='sum')
    kl = 0.5 * torch.sum(mu**2 + sigma2 - 2*log_sigma - 1)
    return recon + kl
```

---

## §8 与 DDPM 的对应

| VAE | DDPM |
|-----|------|
| 单一隐变量 $z$ | $T$ 个隐变量 $x_1, \dots, x_T$ |
| 编码器 $q_\phi(z\|x)$ | **固定**的加噪 $q(x_{1:T}\|x_0)$ |
| 解码器 $p_\theta(x\|z)$ | 反向 $p_\theta(x_{0:T})$ |
| ELBO 训练 | ELBO 训练 |

**DDPM = 链式 VAE**：把"一个隐变量"扩展为"$T$ 个隐变量"，把"学习编码器"改为"固定加噪过程"。

下一份推导手稿 (derive_02) 将处理 DDPM 的具体推导。

---

## §9 自查题

完成本推导后，你应当能不查资料回答：

1. 写出 ELBO 的两种等价形式
2. 证明 $\log p(x) - \mathrm{ELBO} = D_{\mathrm{KL}}(q \| p^{\text{post}})$
3. 推导高斯之间的 KL 闭合形式
4. 解释重参数化的必要性

---

## §10 参考文献

- Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014
- Doersch, *Tutorial on Variational Autoencoders*, 2016
- Bishop, *Pattern Recognition and Machine Learning*, Ch 10
