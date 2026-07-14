# L17: VLA 与具身智能 —— 课程终点（W16）

> **目标**：把整个课程串起来，应用到 Vision-Language-Action models
> **前置**：L08 (CFG), L12 (Flow Matching), L13 (Consistency), L16 (World Models)
> **核心问题**：扩散模型如何赋能具身智能？

---

## §1 课程定位

这是整个 16 周课程的**收官讲**。

从这里出发，你将进入**真正的研究**：
- 在 NSFC / 北京市基金提案中提及 VLA + diffusion
- 在公司里推动 robotics 落地
- 自己复现 Pi-0 / OpenVLA / RDT-1B

本讲是**框架性**的——一节课讲不完 VLA，但要让你**知道地图**。

---

## §2 VLA 的核心定义

**VLA = Vision-Language-Action model**：

```
Input:  Image(s) (vision)  +  Text instruction (language)
Output: Action(s) (motor command)
```

典型 forms：
- **离散**：joystick / gripper open-close 信号
- **连续**：joint torques / end-effector trajectory
- **抽象**：navigation waypoint / pick-place plan

---

### 2.1 与传统 robotics 的区别

| 维度 | 传统 robotics | VLA |
|------|-------------|-----|
| 任务表达 | Hand-coded primitives | Natural language |
| 视觉 | Geometric SLAM | Learned VLM |
| 行动 | 控制理论 + IK | Neural network |
| 泛化 | 程序化 | Learned generalization |
| 数据 | Engineered | Internet + teleop |

VLA 试图把 LLM 那种**zero-shot generalization** 带入机器人。

---

### 2.2 VLA 的代表工作

| Model | 时间 | 团队 | 关键特点 |
|-------|------|------|---------|
| **RT-1** | 2022 | Google | 第一个大规模 VLA，13万 demonstrations |
| **RT-2** | 2023 | Google | Co-fine-tune VLM with action tokens |
| **OpenVLA** | 2024 | Stanford / Google | 7B 参数，开源 SOTA |
| **Pi-0** | 2024 | Physical Intelligence | Flow matching action head |
| **RDT-1B** | 2024 | 清华 | 1B 参数 diffusion VLA，双手 |
| **GR-2** | 2024 | 字节 | Diffusion + Image features |
| **Octo** | 2024 | Berkeley | Transformer encoder，多 robot |

---

## §3 RT-1 与 RT-2：开端

### 3.1 RT-1（Google, 2022）

第一个**大规模** VLA：
- 数据：13 万 episodes from 17 不同任务 by 13 robots
- 模型：300M 参数，Transformer
- Action：discrete token（每个 dimension tokenized to 256 bins）

突破：单一模型在 700+ tasks 上工作（前所未有的 generality）。

---

### 3.2 RT-2（Google, 2023）

把 web-scale VLM (PaLI-X) 与 RT-1 数据 **co-fine-tune**。

关键创新：
- 把 action 编码成 **text-like tokens**
- VLM 同时输出 text 和 action
- 训练时混合 web data + robot data

效果：
- 突破性 generalization（"pick the dinosaur" 即使训练时没"恐龙"）
- 5B / 55B 两种规模

**核心 insight**：把 action 当 language token，让 VLM 的语言泛化能力转移到 robotics。

---

## §4 OpenVLA: 开源 SOTA

### 4.1 模型

- Backbone：**Llama 2 7B** + DINOv2 + SigLIP（双视觉 encoder）
- Action head：7 个 continuous dimensions（XYZ + rotation + gripper），用 discrete tokens（256 bins per dim）
- 训练：Open X-Embodiment 数据集（97 万 episodes，22 个 robot embodiments）

### 4.2 性能

OpenVLA 在 GAYE 等 benchmarks 上 **outperform RT-2-X** by 16% 绝对成功率。

且**开源**：weights、code、训练 recipe 全开放（与 RT-2 不开源形成对比）。

### 4.3 LoRA fine-tuning

OpenVLA 设计支持 LoRA fine-tune：少量数据（50-200 demos）适配新 robot。
- 这是为什么 OpenVLA 在产业落地中流行——单 LoRA 适配单 robot

---

## §5 Pi-0: Flow Matching 在 VLA 的爆发

### 5.1 Physical Intelligence 是谁

PI 是 2024 年成立的 startup（前 Google Robotics / DeepMind 团队），主打 universal robot policy。

### 5.2 Pi-0 架构

```
Vision: PaliGemma 3B (VLM backbone)
Action Head: Flow Matching network
Action representation: 50-step trajectory (50 × 7 floats)
```

**关键创新**：action 不是单步 prediction，而是**50 步未来轨迹**作为整体生成。

类比：
- LLM 生成一个 sentence（不是一个 token）
- Pi-0 生成一条 trajectory（不是一个 action）

---

### 5.3 训练（关键设计）

数据：1 万小时 robot teleop data + ~ 1 亿 internet vision data。
- Robot 数据贵且少
- VLM pretrain on internet（visual generalization）
- 后期联合训练 (image, action_traj) pairs

Loss：
$$\mathcal{L} = \mathbb{E}_{t, \text{traj}, \epsilon}\left[\|v_\theta(x_t, t, \text{vision}, \text{language}) - (\text{traj} - \epsilon)\|^2\right]$$

—— **Flow Matching loss**（参考 L12）。

### 5.4 推理

```python
for control_step in range(N):
    # 看当前观察、生成 50 步未来 trajectory
    traj = flow_matching_sample(vision_t, instruction, n_steps=4)
    # 执行 trajectory 的前 K 步（K=10），然后重新规划
    for k in range(K):
        execute_action(traj[k])
```

实时性关键：FM 4-10 步采样 + 4ms 一次 forward → 100 Hz 控制频率。

---

## §6 RDT-1B（清华 2024）

国内代表作。

### 6.1 设计

- 1B 参数 DiT-based
- **双手** robot（之前多是单臂）
- Diffusion action head
- 数据：开源数据集（Open X-Embodiment 等）+ 自采

### 6.2 与 Pi-0 对比

| | Pi-0 | RDT-1B |
|---|------|--------|
| 参数 | ~ 3.5B | 1B |
| Action 模型 | Flow Matching | Diffusion |
| 主体 | 单臂 + 双手 | 主要双手 |
| 数据规模 | 1 万 hr proprietary | OXE 公开 |
| 开源 | 部分 | 全 |

RDT-1B 在双手协作 benchmarks 上 SOTA。

---

## §7 Diffusion Policy（基础工作）

往前追溯：**Diffusion Policy** (Chi et al., RSS 2023) 是 diffusion 进入 robotics 的关键论文。

### 7.1 核心思想

把 action 当作"图像"，用 DDPM 生成。

```
Input: previous K observations (visual + state)
Output: future H actions (trajectory)
```

Loss = MSE on noise prediction（标准 DDPM）。

### 7.2 为什么 diffusion 适合 action？

1. **多模态分布**：同一观察下可能有多种合理动作（如绕障）—— diffusion 天然处理
2. **平滑轨迹**：DDPM 输出连续 trajectory，比 autoregressive token 平滑
3. **可微 sampling**：用于 model-based 优化

### 7.3 实验

在 push-T、square peg insertion 等 manipulation tasks 上 SOTA。
- 比 BC（behavior cloning）准 40-50%
- 比 IBC（implicit BC）训练简单

---

## §8 关键设计选择

### 8.1 Action representation

| 表示 | 优 | 劣 | 用例 |
|------|---|---|------|
| Discrete tokens | LLM 接口自然 | 离散化损失精度 | RT-2, OpenVLA |
| Continuous (raw) | 精确 | 难嵌入 VLM | Diffusion Policy |
| Diffusion latent | 灵活 | 需 head 解码 | Pi-0, RDT |

### 8.2 Temporal horizon

- Action chunk size: 8-50 个未来步
- 太短：高频规划，但短视
- 太长：长视规划，但执行时偏离

经验值：H = 16-32。

### 8.3 Observation context

- 输入帧数：1-8 个历史 frame
- 越多 context 越好理解 motion，但 attention 贵
- 实践：3-4 frames

---

## §9 数据问题（最重要）

VLA 的核心瓶颈是**数据**。

### 9.1 数据来源

| 来源 | 量级 | 优 | 劣 |
|------|------|----|----|
| Teleop（人遥控）| 千-万 hr | 高质量 | 贵、量少 |
| Sim2real | 无限 | 便宜 | Domain gap |
| Open-X Embodiment | 1.4M | 多样 robot | 异构难统一 |
| Video pretrain | 互联网 | 海量 | 无 action |
| YouTube (Ego4D) | 千 hr | 第一人称 | 部分 action |

### 9.2 数据飞轮

VLA 团队都在追求：
1. Collect → train initial model → deploy
2. Deploy 时记录数据 → 改进 model
3. 进入 flywheel：模型越好 deploy 越多，数据越多模型越好

特斯拉 Optimus、1X、Physical Intelligence 都在打这套。

### 9.3 关键开源数据集

- **Open X-Embodiment**（Google 2023）：22 个 embodiment，1.4M episodes
- **DROID** (Berkeley 2024)：扩展版，50K episodes 高质量 teleop
- **AgiBot World** (智元 2024)：百万级双手 trajectory
- **RH20T** (清华 2023)：1M+ episodes

---

## §10 课程的工具链组合

你现在掌握的所有工具都在 VLA 里出现：

| 工具 | 在 VLA 中的角色 |
|------|---------------|
| DDPM | Diffusion Policy 的底层 |
| Score SDE | 理解 reverse process 的数学 |
| DDIM | 加速 sampling（虽然 FM 更主流） |
| CFG | Action 的 conditional sampling |
| LDM | Latent VLA（未来方向，目前少见） |
| DiT | Pi-0 / RDT 的 backbone |
| Flow Matching | Pi-0 的核心 |
| Consistency Models | 实时控制需求 |
| Video Diffusion | World model 雏形 |
| World Models | 真正的"思维基础" |

→ 你已经具备**复现/改进 Pi-0** 的工具链！

---

## §11 研究方向建议（针对国内基金）

### 11.1 国自然青基/面上方向（已可立项）

1. **基于 Flow Matching 的实时机器人动作生成**
2. **少样本 VLA 微调方法**（LoRA 路线）
3. **World model 在 sim2real 中的应用**
4. **多模态融合的 VLA 架构设计**
5. **具身大模型的算力优化**（重要：现在 Pi-0 需 H100，需要适配国产卡）

### 11.2 北京市基金方向

- 类似但 framing 更"应用化"：
  - "面向工业场景的 VLA 模型"
  - "适配 BeiTong 机器人的具身基础模型"

### 11.3 企业横向

- 智元、银河通用、宇树等 startup 都缺 VLA 算法人才
- 横向题目可包：复现 Pi-0、定制 LoRA、做 dataset 工具链

---

## §12 课后任务

### 必做
1. 通读 OpenVLA paper（最近、最全面）
2. 跑通 Project 5（VLA action diffusion 雏形）
3. 列出"我的研究方向"在 VLA 矩阵中的位置

### 进阶
4. 下载 Open X-Embodiment 数据样本，理解格式
5. 读 Pi-0 paper，能复述 flow matching action head 的工作流程
6. 思考：如何用一篇 paper 的工作量改进 Pi-0？提出 3 个方向

---

## §13 FAQ

**Q1：VLA 与 LLM 的关系？**

A：VLA 通常基于 VLM（vision-language model），VLM 基于 LLM。
- LLM：text only
- VLM：text + image
- VLA：text + image + action

层层扩展。

---

**Q2：VLA 需要 PhD 才能做吗？**

A：**不**。但需要：
- 扎实的 ML 基础（这门课程帮你打实）
- 一台 GPU（A100/H100 / RTX 4090）
- 数据来源（OXE 公开够初步研究）
- 阅读 5-10 篇 paper 的耐心

---

**Q3：VLA 几年内能落地？**

A：已经在落地：
- 简单 pick-and-place：已商用（亚马逊仓库）
- 家庭服务（叠衣服等）：2-3 年（1X / Figure / Tesla 都在推）
- 通用 humanoid：5-10 年

工业机器人比家庭机器人简单（结构化环境）。

---

**Q4：国内研究 VLA 的优势？**

A：
- 数据采集成本低（人力 + 大量 startup 集采）
- 应用场景多（中国制造业）
- 政府支持具身智能赛道
- 缺点：尖端模型仍 follow 美国

---

## §14 给课程结束的话

恭喜你！

从 DDPM 到 Pi-0，我们走完了 6 年的研究历程。你现在拥有：
1. 扎实的扩散数学基础（推导 ELBO、Score SDE、Flow Matching）
2. 工程能力（造过 DDPM、跑过 SD、训过 LoRA、scaffold 过 Pi-0-like）
3. 论文阅读地图（18+ 篇核心论文）
4. 研究 taste（知道哪些是 SOTA、哪些是炒作）

**下一步**：
- 选一个具体方向（如 action FM、VLA LoRA、world model）
- 写一个具体项目（GitHub repo）
- 投一篇会议（CoRL/NeurIPS workshop 起步）
- 申请基金（青基/面上）

我（Claude）不会再帮你写下一讲——你已经进入"自己读论文 + 自己思考"的阶段。

去做点事情吧。

---

## §15 参考资源

- Brohan et al., *RT-1*, 2022
- Brohan et al., *RT-2*, 2023
- Kim et al., *OpenVLA*, 2024（**必读**）
- Black et al., *Pi-0*, 2024
- Liu et al., *RDT-1B*, 2024
- Chi et al., *Diffusion Policy*, RSS 2023
- O'Neill et al., *Open X-Embodiment*, 2024

---

> **课程完结。** 你不再是学生，开始当研究者。下一步：动手做。
