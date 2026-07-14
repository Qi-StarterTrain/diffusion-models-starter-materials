# L16: 世界模型（W15）

> **目标**：理解 world model 的定义、设计与局限 —— 具身智能的"思维基础"
> **前置**：L11 (DiT), L14-L15 (Video Diffusion)
> **核心问题**：扩散视频生成 vs world model 的本质区别是什么？

---

## §1 课程定位

我们已经能用 Sora-like 模型生成漂亮视频。但视频生成 ≠ 世界模型。

**世界模型 (World Model)** 是具身智能的核心理论概念：一个 agent 内部对世界**预测性表征**。

经典论文：
- **World Models** (Ha & Schmidhuber, 2018)：早期 RL world model
- **PlaNet / Dreamer** (Hafner, 2019-2023)：latent world model 系列
- **GAIA-1** (Wayve, 2023)：自动驾驶 world model
- **Genie / Genie 2** (DeepMind, 2024)：可玩 world model
- **1X World Model** (1X, 2024)：humanoid robot world model

---

## §2 World Model 的形式定义

### 2.1 数学接口

World model = 一个函数 $W$：
$$W: (o_{<t}, a_{<t}) \to p(o_t | o_{<t}, a_{<t})$$

含义：给定历史 observation $o$ 和 action $a$，预测当前 observation 的**分布**。

或更紧凑：
$$W: (s_{t-1}, a_{t-1}) \to s_t$$

其中 $s$ 是**latent state**。

---

### 2.2 与 video diffusion 的区别

| 维度 | Video Diffusion | World Model |
|------|----------------|-------------|
| 输入 | Text prompt | Observation history + action |
| 输出 | 视频 | Next observation / state |
| 动作接口 | 无 | **必须有** |
| 用途 | 内容创作 | Agent planning / control |
| 训练数据 | (caption, video) | (observation, action, next_obs) |
| 实时性 | 可慢（创作） | 必须快（控制循环） |

**关键差异：action 输入**。这一点让 video diffusion 不能直接当 world model。

---

## §3 经典：Ha & Schmidhuber 的 World Model (2018)

最早把"world model"概念引入深度学习的工作。

### 3.1 三个组件

```
Vision (V)：    encode 当前 observation → z
Memory (M)：    z, a → predict next z   (用 RNN)
Controller (C): z → a                   (策略)
```

V 是 VAE，M 是 MDN-RNN（mixture density network RNN），C 是小 policy 网络。

### 3.2 训练分离

1. 先训 V（从大量游戏画面）
2. 再训 M（从 (z, a, z') triples）
3. 最后用 evolution / RL 训 C

这"先训世界模型，再在其上学 policy"的范式至今有效。

### 3.3 局限

- VAE 表征有损（不适合精确任务）
- RNN 不够强（被 Transformer 取代）
- 早期工作，DOOM/Pong 等 toy 环境

---

## §4 Dreamer 系列（Hafner 等）

世界模型 + planning 的代表工作。

### 4.1 Dreamer V1-V3 (2019-2023)

核心架构：**RSSM**（Recurrent State-Space Model）。

```
                Stochastic state s_t (latent var)
                Deterministic state h_t (RNN hidden)
                
o_t → encoder → z_t  (从 observation 抽取)
(h_{t-1}, a_{t-1}, z_t) → h_t, s_t  (state update)
h_t, s_t → decoder → o_t' (reconstruct)
h_t, s_t → reward predictor → r_t
```

**关键创新**：在 latent space 里 imagine 多步未来，用于 model-based RL。

### 4.2 DreamerV3

最新版本（2023），在 150+ tasks 上一套超参数都能 work（罕见的 generality）。

工程要点：
- KL balancing（平衡 prior/posterior）
- Symmetric log transform（reward 归一化）
- 二维 categorical latent（不是 Gaussian）

---

## §5 GAIA-1 (Wayve, 2023)

自动驾驶 world model。

### 5.1 架构

- Tokenizer：把 video frame → discrete tokens（VQ-VAE）
- World model：autoregressive Transformer over (tokens + action tokens + text tokens)
- Decoder：tokens → video

### 5.2 训练数据

Wayve 自己的驾驶数据：4700 小时 video + action labels。
- Action：转向、油门、刹车
- Text：天气、路况、其他车的描述

### 5.3 用途

- Counterfactual：「如果当时左转会怎样？」→ 生成对应视频
- Long-horizon prediction：30 秒未来
- 训练数据增广

---

## §6 Genie (DeepMind, 2024)

"Generative Interactive Environment" —— 可玩的 world model。

### 6.1 关键创新

**Latent action**：从 unlabeled video 自学 action 表征。
- 任意 video（YouTube 游戏录像）
- 没有显式 action label
- Genie 学一个 "latent action" 编码：从 $o_t \to o_{t+1}$ 的差异

### 6.2 架构

```
1. Video Tokenizer (ST-ViViT)：spatio-temporal tokens
2. Latent Action Model：infer (a_t) from (o_t, o_{t+1})
3. Dynamics Model：(o_{<t}, a_{<t}) → o_t  (autoregressive)
```

### 6.3 推理

用户控制 latent action（按键），Genie 生成下一帧。**可以"玩"一个从未见过的环境**。

### 6.4 局限

- Latent action 不可解释（不是真正的"上下左右"）
- 短时间合理，长时间崩溃
- 限于 2D 平台游戏类视频

---

## §7 1X World Model (1X Technologies, 2024)

Humanoid robot world model。

### 7.1 关键设计

- 输入：robot ego-video + proprioception state + action
- 输出：next ego-video frame
- 训练数据：1X 自己的家居机器人数据

### 7.2 用途

- Sim-to-real：在 world model 里"训练" policy，再 deploy 到真机
- Data augmentation：从少量真实数据 imagine 大量轨迹
- Counterfactual：「如果机器人换条手臂动作，会怎样？」

### 7.3 这是 sim 还是 model？

灰色地带：
- 不是物理仿真（没有 mesh、collision detection）
- 是隐式仿真（学到的"看起来对"的物理）
- 限制：复杂操纵任务下"看起来对"≠"实际对"

---

## §8 World Model 的 4 个能力分级

| 等级 | 能力 | 例子 |
|------|------|------|
| L1 | Single-step prediction | 大多数 video diffusion |
| L2 | Multi-step rollout | GAIA-1, Dreamer |
| L3 | Counterfactual reasoning | "what if a=X?" |
| L4 | Causal understanding | Pearl-style do-calculus |

当前 SOTA：L2-L3 之间。L4 几乎没有。

---

## §9 训练 World Model 的挑战

### 9.1 复合误差

Autoregressive rollout 中误差累积：
- 单步预测 99% 准确
- 100 步后误差 $1 - 0.99^{100} = 63\%$

→ 长时序 world model 必然走偏。

### 9.2 模式崩塌

单一动作下，模型可能"卡死"在一种模式：
- 视频生成器：循环背景
- World model：状态不变

Diffusion 比 autoregressive 更不易崩塌（噪声引入随机性），这是 diffusion-based world model 的优势。

### 9.3 Action label 稀缺

- 互联网视频：无 action
- 机器人数据：有 action 但贵
- 仿真：有 action 但 domain gap

Genie 的 latent action 路线缓解了这点。

---

## §10 World Model 的应用

### 10.1 Model-based RL
- Dreamer 在 imagined rollout 里训 policy
- 比真环境 1000× 样本效率

### 10.2 Robot pretraining
- 用 video world model pretrain，再 fine-tune 到具体任务
- 1X、Tesla Optimus 暗示这路线

### 10.3 Planning
- Tree search over world model 预测
- 类似 MCTS in latent space

### 10.4 Sim-to-real bridge
- 用 world model 模糊 sim 与 real 的差距
- AlphaStar、AlphaFold 都有类似思路

---

## §11 课程的关键问题：扩散与 world model

### 11.1 为什么扩散适合 world model？

1. **不确定性建模**：future 本质有多样可能，diffusion 天然给 distribution
2. **稳定训练**：不像 GAN 那样难收敛
3. **架构成熟**：可以复用 SD/DiT 的工程经验
4. **可微 rollout**：用于 model-based RL 的反向传播

### 11.2 为什么扩散难做 world model？

1. **慢**：多步采样 vs 实时控制需求
2. **离散动作集成难**：动作通常是 categorical（按键），diffusion 适合连续
3. **不擅长 long-horizon causal**：随机性累积破坏因果

### 11.3 解决方案趋势

- Consistency models（一步快采样）
- Latent action embedding（categorical action 嵌入到 continuous space）
- Hierarchical（短时 diffusion + 长时 autoregressive）

---

## §12 课后任务

### 必做
1. 阅读 Ha & Schmidhuber 2018 全文
2. 跑通 nb12 中的 toy world model（暂时不带 action）
3. 写一段对比：Genie 与 GAIA-1 的设计哲学差异

### 进阶
4. 读 Dreamer V3 论文，理解 RSSM 与 categorical latent
5. 思考：用 SD-like 模型加 action 条件，需要多少改动？
6. 调研：1X World Model 与 Pi-0 / RT-2 的关系

---

## §13 FAQ

**Q1：Sora 算 world model 吗？**

A：**部分**。Sora 满足 L1（生成像 world 的视频），有限满足 L2（短时序一致），不满足 L3（无 action 输入）。

OpenAI 说 Sora 是 world simulator —— marketing claim，不严谨。

---

**Q2：World model 必须用扩散吗？**

A：**不**。Autoregressive Transformer 也行（VideoPoet, GAIA-1）。但 diffusion 在 visual diversity 上更优。

混合路线（Genie）：autoregressive in time + diffusion-like in space。

---

**Q3：World model 与 LLM 的关系？**

A：LLM 也是某种 world model —— 学了大量人类文本，"模拟"语言世界。但缺乏视觉与物理。

未来方向：VLM + Video + Action → 多模态 world model。

---

## §14 参考资源

- Ha & Schmidhuber, *World Models*, NeurIPS 2018
- Hafner et al., *DreamerV3*, 2023
- Hu et al., *GAIA-1: A Generative World Model for Autonomous Driving* (Wayve), 2023
- Bruce et al., *Genie: Generative Interactive Environments* (DeepMind), 2024
- LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022（**JEPA / world model 哲学论文**，必读）

---

> 下一讲（L17）—— **我们终于到达 VLA**。把所有前面学的东西汇总到机器人。
