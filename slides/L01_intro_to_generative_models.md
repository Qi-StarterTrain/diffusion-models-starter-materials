# Lecture 01｜生成模型导论：从 GAN/VAE/Flow 到 Diffusion

> **本讲目标**
> - 理解"生成模型"到底在解决什么问题
> - 掌握 GAN、VAE、Flow、Diffusion 四类方法的本质区别
> - 建立"为什么 Diffusion 后来居上"的直觉
> - 形成阅读生成模型论文的统一视角

---

## §1 生成模型要解决的问题

### 1.1 一个直观的例子

给你 1 万张猫的照片。能否让算法**生成出新的、看起来像真猫但又不是训练集中任何一张的猫**？

这就是**生成建模**（generative modeling）。

---

### 1.2 形式化定义

我们假设观察到的数据 $x_1, x_2, \dots, x_N$ 来自某个**未知的真实分布** $p_{\text{data}}(x)$。

生成模型的目标：**学习一个参数化分布 $p_\theta(x)$，使得它逼近 $p_{\text{data}}$**。

学到 $p_\theta$ 后，我们可以：
1. **采样**：生成新数据 $x \sim p_\theta(x)$
2. **评估**：给定一个 $x$，计算其 likelihood（不是所有方法都支持）
3. **条件生成**：扩展为 $p_\theta(x | y)$，例如文本生成图像

---

### 1.3 核心挑战

为什么生成建模困难？

**挑战 1：维度灾难**

一张 256×256 RGB 图像生活在 $\mathbb{R}^{196608}$ 空间。但实际有意义的图像只占其中极小子流形。直接对这个空间建模几乎不可能。

**挑战 2：分布的复杂性**

$p_{\text{data}}(x)$ 通常是**高度多模态**的。一张猫图、一张狗图、一张山的图，可能在像素空间相距很远，但都是"真实图像"——分布有很多个 mode。

**挑战 3：评估困难**

我们看不见真实分布，只看到样本。一个生成模型"好不好"很难量化（FID、IS 都是近似指标）。

---

## §2 四大流派：从设计哲学看差异

下面用一个统一的框架来理解四类生成模型。

> **统一视角**：所有生成模型都需要一个**简单分布 $p(z)$**（如标准高斯）和一个**变换 $z \mapsto x$**（学到的）。差异在于"变换怎么定义、怎么学"。

---

### 2.1 GAN（Generative Adversarial Networks）

**核心思想**：训练一个生成器 $G$ 把噪声映射到图像，再训练一个判别器 $D$ 区分真假，让两者博弈。

**采样**：
$$z \sim \mathcal{N}(0, I), \quad x = G_\theta(z)$$

**训练目标**（minimax 博弈）：
$$\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_z[\log(1 - D(G(z)))]$$

**优点**：
- 生成质量高（视觉效果好）
- 采样快（一次前向）
- 模型架构灵活

**缺点**：
- ❌ 训练不稳定（mode collapse、判别器过强）
- ❌ 不能直接计算 likelihood
- ❌ 难以做条件控制（要重训）

---

### 2.2 VAE（Variational Autoencoder）

**核心思想**：用变分推断学习数据的隐变量表示，通过 ELBO 最大化对数似然下界。

**采样**：
$$z \sim p(z), \quad x \sim p_\theta(x | z)$$

**训练目标**（ELBO）：
$$\mathcal{L}_{\mathrm{ELBO}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))$$

**优点**：
- ✅ 训练稳定
- ✅ 有 likelihood 下界
- ✅ 隐空间结构良好（可插值、可编辑）

**缺点**：
- ❌ 生成图像偏模糊（高斯解码器的本质局限）
- ❌ 后验坍塌（posterior collapse）

---

### 2.3 Flow-based Model

**核心思想**：构造一系列**可逆**变换 $f_1, f_2, \dots, f_K$，让 $x = f_K \circ \cdots \circ f_1(z)$。利用换元公式直接计算 likelihood。

**采样**：
$$z \sim p(z), \quad x = f_\theta(z)$$

**Likelihood 计算**：
$$\log p_\theta(x) = \log p(z) + \log |\det J_{f^{-1}}(x)|$$

**优点**：
- ✅ 可计算精确 likelihood
- ✅ 可逆（编码、解码、似然都直接）

**缺点**：
- ❌ 网络结构受限（必须可逆且雅可比可计算）
- ❌ 表达能力不如 GAN
- ❌ 参数量大、训练慢

---

### 2.4 Diffusion Models（本课程主角）

**核心思想**：把生成过程拆成**很多步小的去噪**。先定义一个固定的"加噪过程"把数据变成噪声，再让网络学习"反过程"。

**前向过程（固定，不学习）**：
$$x_0 \to x_1 \to x_2 \to \cdots \to x_T$$
每步加一点高斯噪声，最终 $x_T$ 接近纯噪声。

**反向过程（学习）**：
$$x_T \to x_{T-1} \to \cdots \to x_0$$
学一个网络 $\epsilon_\theta(x_t, t)$ 预测每步加的噪声，再反向去掉。

**采样**：从 $x_T \sim \mathcal{N}(0, I)$ 出发，迭代去噪 $T$ 步得到 $x_0$。

**优点**：
- ✅ 训练**极其稳定**（就是 MSE 回归！）
- ✅ 生成质量高（已超过 GAN）
- ✅ 可以无缝条件控制（CFG、ControlNet）
- ✅ 数学上有 ELBO 保证
- ✅ 可扩展到 video、3D、机器人动作

**缺点**：
- ❌ 采样慢（早期 1000 步，现已 1 步可达，靠 distillation）
- ❌ 训练数据需求大

---

## §3 直观对比：四种方法的"形状"

```
GAN：    z ──[Generator]──> x
         （一步映射，靠对抗博弈训练）

VAE：    x ──[Encoder]──> z ──[Decoder]──> x'
         （编码后解码，最小化重构 + KL）

Flow：   z ──[f1]──>──[f2]──>──[f3]──> x  （可逆链）
         x ──[f3⁻¹]──>──[f2⁻¹]──>──[f1⁻¹]──> z  （反向也可逆）

Diffusion: x_0 ──>──>──>──>──> x_T  （固定加噪，类似随机游走）
           x_0 <──<──<──<──< x_T  （学一个去噪网络反向走）
```

**核心差异**：

| 方法 | 学什么 | 一步还是多步 | Likelihood |
|------|--------|--------------|------------|
| GAN | 生成器 + 判别器 | 一步生成 | ❌ |
| VAE | 编码器 + 解码器 | 一步生成 | 下界 |
| Flow | 可逆变换 | 多层但形式受限 | ✅ 精确 |
| Diffusion | 单一去噪网络 | 多步迭代 | 下界（ELBO） |

---

## §4 为什么 Diffusion 后来居上？

回顾历史：
- 2014：GAN 提出，统治图像生成 6–7 年
- 2015：Diffusion 雏形（Sohl-Dickstein），但效果远落后于 GAN
- 2020：DDPM（Ho et al.）实现质量飞跃
- 2021：Diffusion 在 ImageNet 上击败 GAN（Dhariwal & Nichol）
- 2022：Stable Diffusion 普及，引爆 AIGC
- 2023+：成为视频、3D、机器人动作建模的事实标准

**Diffusion 胜出的三个根本原因**：

### 4.1 训练稳定性

GAN 的 minimax 博弈极其难调，一旦失衡就 collapse。
Diffusion 的训练目标是**纯粹的 MSE 回归**——和训练一个分类器一样稳定。

### 4.2 任务分解的智慧

GAN 让网络一次性把噪声变图像——这是极困难的"一锤定音"。
Diffusion 把生成拆成 $T$ 步小问题，每步只需"去掉一点点噪声"——每个子问题简单。

> **类比**：GAN 像"一笔画出蒙娜丽莎"；Diffusion 像"用 1000 笔逐步描出蒙娜丽莎"。

### 4.3 灵活的条件接口

文本到图像、图像到图像、inpainting、ControlNet、LoRA 微调——所有这些**都不需要重新设计模型架构**，只要在训练或采样阶段加上条件即可。

GAN 做条件生成往往要重新训练 conditional GAN。

---

## §5 一个常见的误解

**误解**：Diffusion 的多步采样是它的本质特征，去掉就不是 Diffusion 了。

**事实**：
- DDIM 已经把采样从 1000 步降到 50 步
- Consistency Model 可以做到**1 步生成**
- 即使是单步生成，"扩散思想"——通过定义加噪过程并学习反过程——仍然是模型的核心

所以：**Diffusion 的本质不是"多步"，而是"通过加噪—去噪学习生成"**。这个范式可以与各种采样加速兼容。

---

## §6 Diffusion 的研究地图（提前预告）

后续 13 周我们会沿着下面这条主线展开：

```
W2-W4:   DDPM 与 Improved DDPM
            ↓ "怎么训练？怎么采样？"
W5:      Score SDE
            ↓ "DDPM 与 score matching 的统一"
W6:      DDIM, DPM-Solver
            ↓ "怎么加速采样？"
W7:      CFG（无分类器引导）
            ↓ "怎么做条件生成？"
W8-W9:   LDM / Stable Diffusion + LoRA / ControlNet
            ↓ "怎么扩展到高分辨率与可控生成？"
W10:     DiT
            ↓ "U-Net 已经过时？"
W11-W12: Flow Matching, Consistency Models
            ↓ "下一代范式与极致加速"
W13-W14: Video Diffusion, World Model, Diffusion Policy
            ↓ "扩展到时间维度与具身智能"
```

每一步都是对前面"未解决问题"的回应。读完整条线，你会获得一种**理解新论文的"地图感"**——任何新工作大致都能放进这张图的某个位置。

---

## §7 本讲核心要点回顾

1. 生成模型的目标是逼近 $p_{\text{data}}(x)$，能采样、能评估
2. GAN/VAE/Flow/Diffusion 各有设计哲学，差异主要在"如何参数化变换 + 如何训练"
3. Diffusion 后来居上的根本原因：**训练稳定 + 任务分解 + 条件灵活**
4. Diffusion 的本质是"通过加噪—去噪学习生成"，与采样步数无关
5. 我们将沿一条历史—技术混合的主线，理解 Diffusion 从经典到前沿的演化

---

## §8 课后任务

### 必做

1. **阅读笔记**：阅读 Sohl-Dickstein 2015《Deep Unsupervised Learning using Nonequilibrium Thermodynamics》第 1–3 节，按 reading note 模板提交一页笔记。重点理解**为什么 forward process 可以固定、reverse process 需要学习**。

2. **数学前置自测**：完成 `self_assessment_quiz.md`，对照答案自查，记录哪些题答错及原因。

3. **环境搭建**：
   - 安装 Python 3.10+、PyTorch 2.0+、CUDA
   - 安装 wandb，注册账号，跑通 `wandb.init()`
   - 跑通 PyTorch 官方 60-minute blitz 中的 MNIST 训练

### 选做（推荐）

4. **思考题**（写 200 字短文）：如果你要向一个完全不懂深度学习的朋友解释"什么是 Diffusion 模型"，你会怎么类比？请写下你的版本，下次课分享。

5. **预习**：阅读 Lilian Weng 博客 *What are Diffusion Models?* 前半部分，对加噪过程建立直觉。

---

## §9 推荐进一步阅读

| 资源 | 类型 | 难度 | 重点 |
|------|------|------|------|
| Goodfellow 《Deep Learning》Chapter 20 | 教材 | ⭐⭐ | 经典生成模型综述 |
| Lilian Weng 博客 *From Autoencoder to Beta-VAE* | 博客 | ⭐⭐ | VAE 直觉 |
| GAN tutorial（Goodfellow NeurIPS 2016） | 视频 | ⭐⭐ | GAN 入门权威 |
| OpenAI 博客 *Generative Models* | 博客 | ⭐ | 高层次综述 |
| Lilian Weng 博客 *What are Diffusion Models?* | 博客 | ⭐⭐⭐ | Diffusion 入门首选 |

---

## §10 常见问题

**Q: 我现在能跑通 PyTorch 标准 MNIST 训练，但没接触过 GAN/VAE，能跟上这门课吗？**

A: 完全可以。本课程从 DDPM 开始**自包含地**讲解 Diffusion 体系，不需要熟悉 GAN/VAE 的细节。但建议你在 W1 结束前快速浏览 VAE 的核心思路（特别是 ELBO），因为 DDPM 在数学上和 VAE 是同一族。

**Q: 高斯分布、KL 散度、ELBO 我都不太熟，要先去补吗？**

A: 优先做完 `self_assessment_quiz.md`。如果错了 4 题以上，**强烈建议**先花 2–3 天读完 `prob_review.md`。后面的 DDPM 推导对这些工具的依赖度极高，基础不牢会非常痛苦。

**Q: 为什么这门课不直接讲 Stable Diffusion，而要从 DDPM 开始？**

A: Stable Diffusion 是工程上的精妙组合，但其每个组件（U-Net、噪声预测、CFG、VAE 压缩、cross-attention）都源自更基础的论文。如果跳过基础直接看 SD，你会知道"怎么用"但不知道"为什么这样设计"。本课程的目标不是"会用 diffusers"，而是"看到下一篇 SD 改进论文能立刻理解 novelty 在哪里"。
