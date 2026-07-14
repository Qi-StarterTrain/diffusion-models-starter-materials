# L15: Sora 解剖 — 视频生成模型作为世界模拟器（W14）

> **目标**：深入 Sora 的设计哲学与工程细节
> **前置**：L11 (DiT), L14 (Video Diffusion)
> **核心问题**：Sora 的"涌现物理"是真理解还是模式匹配？

---

## §1 课程定位

Sora（2024/2 公布，2024/12 发布）是 OpenAI 的视频生成系统：
- 60 秒 1080p 视频
- 单 prompt 多种风格
- "World simulator" 营销词

但 Sora **没有详细公开论文**——只有 technical report 和官方博客。本讲基于：
1. 官方 Technical Report (2024/2)
2. 业界逆向分析（DiT-style 架构）
3. 开源复现：Open-Sora、CogVideoX、Mochi 等

> **批判性阅读**：Sora technical report 是 marketing-heavy 的，要分清"OpenAI claims" 与"实际能做"。

---

## §2 官方 Technical Report 关键信息

### 2.1 标题
*"Video generation models as world simulators"*

—— 自我定位为"世界模拟器"，暗示具身/物理意义。

### 2.2 核心技术声明

1. **Diffusion Transformer (DiT) backbone** —— 不是 UNet
2. **Spacetime patches** —— 把视频拆成时空 patch tokens
3. **Variable resolutions, durations, aspect ratios** —— 任意分辨率与时长
4. **Recaptioning** —— 用 GPT-4V 给训练视频生成详细 caption

每点都值得展开。

---

## §3 Spacetime Patches

### 3.1 核心想法

视频 $\in \mathbb{R}^{T \times H \times W \times C}$
→ 3D patches of size $(p_t, p_h, p_w)$
→ tokens $\in \mathbb{R}^{N \times D}$

其中 $N = (T/p_t)(H/p_h)(W/p_w)$。

例：60s × 1080p × 30FPS = $1800 \times 1920 \times 1080 \times 3$
- patch $4 \times 32 \times 32$: $N = 450 \times 60 \times 33.75 = 911,250$ tokens
- 完整 attention $O(N^2) \approx 10^{12}$ → 不可行
- 必须用 sparse/window attention

---

### 3.2 与朴素 video DiT 区别

朴素：在 latent space 做 patchify（已经 8x 下采样）
Sora：可能用更激进的下采样

具体 patch size 未公开，但从 demo 时长与显存推算：可能用 $(p_t, p_h, p_w) = (4, 16, 16)$。

---

### 3.3 任意分辨率/时长的实现

DiT 天然支持变 N（attention 不限长度）。Sora 的关键：
- Patch tokenizer 不固定 (T, H, W)
- Position embedding 用 **2D / 3D rotary** 或 **continuous (factorized)**

实践中：
```python
# Pseudo
positions_3d = torch.stack([
    torch.arange(T).repeat_interleave(H * W),
    torch.arange(H).repeat(T).repeat_interleave(W),
    torch.arange(W).repeat(T * H),
], dim=-1)
pos_emb = RotaryEmbedding3D(positions_3d)
```

---

## §4 Re-captioning（关键）

### 4.1 问题

互联网视频的 caption 通常很短或不存在。模型学不到细致的 prompt understanding。

### 4.2 Sora 的解决

1. 训练一个"highly descriptive captioner"（疑似 GPT-4V）
2. 对训练集每个视频生成长详细 caption
3. 用这些 caption 训 Sora

这与 **DALL-E 3** 的做法一致——"prompt quality" 是性能的根本来源之一。

### 4.3 GPT-4 expansion at inference

推理时：
- 用户输入短 prompt
- GPT-4 自动 expand 成长 caption
- Sora 用 long caption

所以用户看到的"prompt following"质量部分来自 GPT-4 expansion，不全是 Sora 自己。

---

## §5 Sora 的"涌现"现象

Tech report 列举：
1. 3D consistency
2. Long-range coherence and object permanence
3. Interaction with the world（物体相互作用）
4. Simulating digital worlds（游戏画面）

这些不是显式训练的，而是**大规模训练涌现**。

---

### 5.1 3D Consistency

模型生成的"3D视角变化"看起来合理（相机绕物体走能看到不同角度）。

**怀疑论**：这真的是 3D 理解吗，还是大量类似训练数据的拼凑？

证据点：
- 失败案例多：相机一旦做复杂运动，3D 一致性破坏
- 没有显式 3D representation
- 类似训练数据的"内推" vs 真正"外推"

---

### 5.2 Object Permanence

物体被遮挡后重新出现仍是同一物体——这是经典认知科学 benchmark。

Sora 在**短时间**（<5s）下表现不错，**长时间**仍有失败。

---

### 5.3 物理一致性

- 杯子掉地：经常不破（违反物理）
- 流体：表面合理但深度运动诡异
- 人手：手指数错（与图像 SD 同病）

→ Sora 像"高级模式匹配器"，对常见物理凑合，复杂情况经常失败。

---

## §6 Sora 的训练规模（估算）

未公开，但业界估算：
- **数据**：上千 PB（疑似 YouTube + 自有视频）
- **算力**：1-2 万 H100 卡训 1-3 个月
- **参数**：3B - 10B（小于 GPT-4 但大于 SD XL）

**Cost 估算**：$50M - $200M 训练成本。这是为什么 Sora 不开源——economic moat。

---

## §7 Sora 的局限（官方承认）

Tech report 末尾承认：
1. **物理交互**：玻璃杯/液体的物理不准
2. **因果关系**：可能"反因果"（如先洞后挖）
3. **复杂运动**：长时间复杂运动失败
4. **多对象**：5+ 个独立运动对象就乱套

→ Sora **不是真正的世界模型**，是**"看起来像"的视频生成器**。

---

## §8 开源复现

### 8.1 Open-Sora (HPC-AI)

中国团队，2024 年陆续放出：
- 1.0: 16 frames @ 512×512，约 1.5B params
- 2.0: 改进 patch size、训练策略
- Github 开源

性能：明显不如 Sora，但在学术研究上够用。

### 8.2 CogVideoX (Zhipu AI, 2024)

清华系，开源 video DiT：
- 2B / 5B 两个版本
- 6s @ 720×480
- 支持 text-to-video 和 image-to-video

### 8.3 Mochi 1 (Genmo, 2024)

10B 参数的 video DiT，开源 weights：
- 5.4s @ 480p
- 在公开 benchmarks 上是 open-source SOTA

### 8.4 LTX-Video (Lightricks, 2024)

2B 参数轻量化，2 秒生成 24fps 30 帧：
- 强调速度而非质量
- 适合移动端部署

---

## §9 Sora 工程要点（推测）

### 9.1 Token reduction

完整 attention $O(N^2)$ 在 N=1M 不可行。可能用：
- **Spatial-Temporal factorization**：先 spatial attn 再 temporal
- **Window attention**：每个 token 只看附近时空 window
- **Hierarchical**：早期 layer 用 local，深 layer 用 global

### 9.2 Patch sliding (caching)

短时窗滑动：
- 生成 frames [0, 32]
- 滑动到 [16, 48]：复用 [16, 32] 的 KV cache，新算 [33, 48]
- 长视频靠这个 sliding window 完成

### 9.3 Quality control loop

业界传闻：Sora 内部有 quality scorer，subpar 输出会自动重试。这部分官方未确认。

---

## §10 与 VLA / Embodied 的连接（**重点**）

### 10.1 Sora 是世界模型吗？

**部分**。视频生成 = 隐式世界模型。
- 缺点：没有 action 输入接口
- 缺点：没有 state representation
- 优点：视觉一致性强，可作为 backbone

### 10.2 Action-Conditioned Video Diffusion

把 Sora-like 模型加 action 输入：
```
Input: prev frames + action sequence
Output: future frames
```

代表：
- **GAIA-1** (Wayve, 2023)：driving world model
- **Genie** (DeepMind, 2024)：动作条件视频
- **WorldDreamer / 1X World Model**：通用具身世界模型

### 10.3 Sora 作为 VLA 的"输出端"？

想象：
- Vision-Language Model 接收 instruction
- Sora-like decoder 生成 future video
- 从生成的视频"读出" action

这是**反向**的世界模型用法。Pi-0 等没这么做，但 Wayve、Tesla AI Day 都暗示类似方向。

---

## §11 课后任务

### 必做
1. 通读 Sora technical report（公开版，约 30 min）
2. 用 Open-Sora 或 CogVideoX 生成 5 个对比视频
3. 分析 Sora demo 中 3 个失败案例（OpenAI 自己放的）

### 进阶
4. 读 Open-Sora 源码，对比与你想象中的 Sora 差距
5. 提出一个"加 action 条件"的设计 sketch
6. 思考：为什么 Sora 不公开训练细节？（商业 vs 学术）

---

## §12 FAQ

**Q1：Sora 是 GPT-4-of-video 吗？**

A：**夸张**。Sora 在视频生成 SOTA，但远不像 GPT-4 在文本上那样的代际跨越。Sora 与 SVD/Pika 的差距 < GPT-4 与 GPT-3 的差距。

---

**Q2：Sora 能算 AGI？**

A：**不能**。Sora 没有：
- 自主目标设定
- 推理与规划
- 与世界交互的能力

它是个**好的视频生成器**，但不是 agent。把"world simulator"标签理解为营销而非技术。

---

**Q3：Sora 的"涌现"是真的吗？**

A：**部分**。
- 视觉质量涌现 ✓
- 短时序一致性涌现 ✓
- 物理理解涌现 ✗（很多失败案例）
- 因果推理涌现 ✗

OpenAI 论文 cherry-picked 漂亮例子，但 X 上有大量 Sora 失败 demo。

---

**Q4：复现 Sora 要多少钱？**

A：业界估算 $10-50M。Open-Sora 团队几百万人民币的版本明显不如。这是 video diffusion 的"高门槛领域"。

---

## §13 参考资源

- OpenAI, *Video generation models as world simulators*（**Sora technical report**）, 2024
- Brooks et al. (OpenAI), various Sora blog posts
- Bar-Tal et al., *Lumiere*, 2024 (Google 同期类似工作)
- HPC-AI Tech, *Open-Sora* (开源复现)
- Yang et al., *CogVideoX*, 2024（**最详细的 video DiT 开源论文**，必读）

---

> 下一讲（L16）我们深入 world model 这个概念——它不只是"高质量视频生成"。
