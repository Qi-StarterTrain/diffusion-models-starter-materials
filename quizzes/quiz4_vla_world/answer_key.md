# Quiz 4 答案钥匙（W15-W16）

> **注意**：参考答案。学生若答出等价或更深刻内容应给满分。

---

## 一、概念题（30 分）

### 1. Video gen vs World Model

**核心区别**：world model 必须有 **action 输入**接口。

- Video generation: 输入 text → 输出视频
- World model: 输入 $(s_{t-1}, a_{t-1}) \to s_t$，可被 agent 用于 planning

**Sora 不是 world model**：
- 无 action 接口
- 无 explicit state representation
- 长 horizon causal 不准（杯子掉地不破）
- 是"看起来像 world simulation 的 video generator"

OpenAI 的 "world simulator" 标签属于 marketing claim。

---

### 2. World model 4 个能力等级

| 等级 | 能力 | 代表 |
|------|------|------|
| L1 | Single-step prediction | 大多数 video diffusion (Sora 短时, Imagen Video) |
| L2 | Multi-step rollout | GAIA-1, Dreamer V3 |
| L3 | Counterfactual ("what if a=X?") | Wayve GAIA-1（部分），Genie |
| L4 | Causal understanding (do-calculus) | **几乎没有**（Pearl-style；研究方向） |

当前 SOTA 在 L2-L3 之间。L4 是 LeCun 的 JEPA 远景。

---

### 3. Pi-0 用 FM 的工程原因

至少 3 点：
1. **少步采样**：FM 4-10 步 vs DDPM 50 步，让 50-100 Hz 控制可行
2. **简洁实现**：linear path + MSE，没有 $\beta$ schedule、$\bar\alpha$ 等复杂数学
3. **连续 action 自然适配**：FM 对连续 trajectory 直接处理，DDPM 通常需要离散化或额外 reparameterization
4. **多模态分布保留**：FM 与 DDPM 都能；但 FM 在 trajectory level 更直观
5. **SD 3 已验证**：image gen 上 FM 优于 DDPM，迁移到 action 风险低

---

### 4. Discrete token (OpenVLA) vs continuous FM (Pi-0)

| | OpenVLA discrete token | Pi-0 continuous FM |
|---|----|----|
| Action 精度 | 量化损失（256 bins） | 连续无损 |
| 推理速度 | 慢（7 个 token 顺序 decode） | 快（FM 4 步 parallel） |
| 与 LLM 集成 | 自然（共享 vocabulary） | 需要独立 head |
| Language grounding | 强（共享 backbone） | 中等（adapter） |
| 多模态分布 | 通过 sampling temperature | 通过 noise prior 自然给 |

**Trade-off**：
- discrete token 牺牲精度，赚到 language grounding 强
- continuous FM 牺牲与 LLM 的统一性，赚到 real-time control

---

### 5. Action chunking

把"未来 $H$ 步 actions"作为一个整体 chunk $\mathbf{a}_{1:H} \in \mathbb{R}^{H \times A}$ 生成（如 H=32-50）。

**为什么不单步**：
1. **Long-horizon planning**：单步预测 myopic；chunking 让模型 "think ahead"
2. **降低 inference 频率**：每 K 步重新规划而非每步，K=10 让 10× 频率适应
3. **Action smoothness**：trajectory 整体生成更平滑（不会突然反向）
4. **多模态保留**：单步 a 易塌缩到平均，trajectory 不易

Chunk size **不能太大**（>100）：
- 长 trajectory 与当前观察的相关性减弱
- 模型参数压力大
- 不可重新规划（如果环境变化）

---

### 6. Diffusion Policy vs BC

**核心优势**：处理 **multi-modal action distribution**。

具体场景：机器人面对桌上一杯水，有两种合理 grasp：
- 从顶部抓（手 vertical）
- 从侧面抓（手 horizontal）

BC (MSE) 会学**两个 action 的平均**——结果是手在中间（45°），实际上**两个都不对**。

Diffusion Policy 通过 stochastic noise 自然给出"要么 vertical 要么 horizontal"的 bimodal 分布。

实证：在 push-T、square-peg insertion 等 task 上 DiPo 比 BC 高 40-50% 成功率。

---

## 二、推导题（40 分）

### 7. Action FM 训练（10 分）

**(a) Loss**：

$$\mathcal{L} = \mathbb{E}_{t, a_{1:H}, \epsilon, v, l}\left[\|v_\theta(a_t, t, v, l) - (a_{1:H} - \epsilon)\|^2\right]$$

其中 $a_t = (1-t) \epsilon + t \cdot a_{1:H}$，$v$ 是 vision feature，$l$ 是 language。

---

**(b) 采样伪代码**：

```python
def sample_action(model, obs, n_steps=4):
    v = vision_encoder(obs['image'])
    l = language_encoder(obs['instruction'])
    a = torch.randn(H, action_dim)  # 从 x_0 = noise 起
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = i * dt
        v_pred = model(a, torch.tensor(t), v, l)
        a = a + v_pred * dt  # Euler step
    return a  # a_{1:H}，可执行的 trajectory
```

---

**(c) 最多采样步数**：

- 50 Hz = 20 ms 控制周期
- Single forward: 4 ms
- 最多 $20 / 4 = 5$ steps
- Pi-0 实际用 4 步（留 margin）

---

### 8. Compound error（10 分）

**(a) 独立错误假设下 100 步全对的概率**：

$$P(\text{all correct}) = 0.99^{100} \approx 0.366$$

—— 约 37%。已经不太行。

---

**(b) 累积错误模型**：

设第 $k$ 步的错误率 $\epsilon_k$ 满足 $\epsilon_k = \epsilon_{k-1} \cdot 1.1$（指数累积）：
$$\epsilon_k = \epsilon_0 \cdot 1.1^k = 0.01 \cdot 1.1^k$$

第 50 步：$\epsilon_{50} = 0.01 \cdot 1.1^{50} \approx 0.01 \cdot 117.4 = 1.174$ → 严重错误（饱和到 1）

第 100 步：$\epsilon_{100} \approx 0.01 \cdot 1.1^{100} = 0.01 \cdot 13780 \approx 138$ → 完全无意义

实际系统会饱和（错误率 cap 在 1），但说明：**100 步后预测几乎完全错**。

---

**(c) 这说明什么**：

1. Autoregressive world model 在长 horizon 下会迅速失败
2. Sora 60 秒视频"看起来对"是因为：
   - 它不是严格的 autoregressive（用 spacetime patches 全局 attention）
   - Visual quality 与 causal consistency 是两件事——前者可由 high-level pattern matching 维持，后者要严格因果链
3. 真正 world model 需要：
   - State 表征的稳定性（不会 drift）
   - Counterfactual reasoning（错了能纠正）
   - 这远比 Sora 难得多

---

### 9. OpenVLA Action 量化（10 分）

**(a) 均匀分布下 RMS 量化误差**：

每 bin 宽度 $\Delta = 2 / 256 = 1/128$。量化误差均匀分布在 $[-\Delta/2, \Delta/2]$。

RMS = $\Delta / \sqrt{12} = (1/128) / \sqrt{12} \approx 2.26 \times 10^{-3}$

—— 每 dim 误差约 0.2%。

---

**(b) Quantile-based 优势**：

如果 action distribution 集中在某个区间（如 90% 的 action 在 [-0.3, 0.3]），等宽 binning 浪费了 [-1, -0.3] 与 [0.3, 1] 区间的 bin。

Quantile-based 使每 bin 覆盖**相同 mass**——常用 action 周围 bin 密集（小误差），罕见 action 周围 bin 稀疏（大误差但被采样几率低）。

**数学**：在 KL divergence 意义下，quantile-based binning 是 maximum-entropy 量化。

直觉：把 bins 当成"采样资源"，集中投放到 mass 高的区域。

---

**(c) 256 → 64 bins**：

RMS 误差与 bin 数成反比（$\Delta \propto 1/N$）：
- 256 bins: 误差 $\approx 2.26 \times 10^{-3}$
- 64 bins: 误差 $\approx 9 \times 10^{-3}$ (~4× 大)

对实时控制：
- 7-DoF arm 操纵：误差 0.1° 量级通常 OK，但 0.9° 在精细 task（如插针）就不可接受
- **Trade-off 不可一概而论**——简单 reach task 64 bins 也 OK，精细任务必须 256+

实践：OpenVLA 选 256 是为了能覆盖各种任务。

---

### 10. Vision encoder（10 分）

**(a) 预训练 vision encoder 的优势**：

从**信息论**角度：
- 预训练（ImageNet, LAION）已经在大规模数据上学到了通用 visual features
- 这些 features 包含 task-relevant 与 task-irrelevant 的混合，但 task-relevant 部分够强
- VLA 训练数据少（万小时 vs ImageNet 14M images），从头训会 underfit
- 预训练给一个**好的初始化**，相当于免费的 representation prior

类比：用 BERT 初始化 NLP task vs 从头训。

---

**(b) DINOv2 + SigLIP 互补**：

- **DINOv2**（自监督）：强 geometry / texture / part-level features，但 language understanding 弱
- **SigLIP**（image-text contrastive）：强 semantic / category-level alignment with text

具体例子：
- "Pick up the red mug"：SigLIP 知道"red mug"语义
- 但要 grasp，需要知道 mug handle 的几何位置 → DINOv2 强

Concat 让两个互补 features 都进入 policy。

—— OpenVLA 实证：用 DINOv2 + SigLIP 比单一 encoder 成功率高 5-10%。

---

**(c) Fine-tune vision encoder？**

**支持 fine-tune**：
- VLA 数据 distribution 与 ImageNet/LAION 不同（机器人 ego-view、特定光照）
- 末端 task-specific features 需要适应
- 模型学到的 visual prior 不变（如果 LR 小）

**反对 fine-tune**：
- 大模型 fine-tune 易 catastrophic forgetting 通用能力
- 数据少时 overfit 风险高
- Frozen encoder 训练快、内存省、稳定

**实践**：通常**部分 fine-tune**——最后几层 unfreeze，早期层 frozen。或者用 **LoRA 加 vision encoder**（极小代价微调）。

---

## 三、实验设计题（30 分）

### 11. Sim-to-real gap 分析（15 分）

**(a) 3 个 hypothesis**：

H1（**视觉 domain gap**）：仿真渲染太"干净"，真机相机有噪声、光照变化、镜面反射等

H2（**物理 domain gap**）：仿真 friction / contact model 与真实物理不同；robot dynamics 误差

H3（**Action 执行 gap**）：仿真 action 是理想化的（瞬时跟踪），真机有 control delay、joint limit、PID 误差

---

**(b) 验证实验**：

H1 验证：
- 在真机上**重新渲染**真实图像（去除特定 visual feature），看 policy 输出是否变化
- 评估指标：在合成 noisy sim image 上的 policy success rate
- 控制变量：保持真实物理状态相同

H2 验证：
- 在仿真中**复现真机失败**：从真机 trajectory 起始，用 sim dynamics rollout 应该到达同样状态，看是否偏离
- 评估指标：trajectory MSE between sim rollout 与 real rollout
- 用 system identification 工具（如 MuJoCo 参数 fitting）

H3 验证：
- 把真机的 action 命令记录下，在 sim 中重放，看是否到达相同 state
- 评估指标：state divergence over time
- 加入控制 delay simulation 看效果

---

**(c) 解决方案 priority**：

通常 priority：
1. **H1（视觉 gap）**：最易解决（domain randomization 在 sim 训练时加 visual aug）
2. **H3（action gap）**：中等（加 dynamics randomization 或 system identification）
3. **H2（物理 gap）**：最难（contact model 几乎不可能完美 sim）

策略：**Sim with randomization + a few real-world fine-tune**。常见做法：
- Sim 训练 1B steps with massive domain randomization
- 真实 100 demos LoRA fine-tune
- 最终成功率 70-85%（不会完美，但实用）

---

### 12. NSFC 青基实验设计（15 分）

**Experiment 1：FM 速度证明（最小可行）**

- 目标：在 LIBERO benchmark（10 个 manipulation tasks）上证明 FM action 比 DDPM 快 5×，成功率相当
- Setup：固定 Diffusion Policy 架构，替换 action head 为 DDPM (50 steps) vs FM (10 steps) vs FM (4 steps)
- 评估：每 task 100 episodes，记录成功率与 inference latency
- 预期：FM 4 steps 与 DDPM 50 steps 成功率差 < 5%，inference 快 10×

**Experiment 2：OpenVLA + FM 适配（生态）**

- 目标：把 OpenVLA 的 discrete token head 改造为 FM continuous head，做完整 ablation
- Setup：
  - Backbone 不变（OpenVLA 的 Llama 2 + DINOv2 + SigLIP）
  - 替换 action head: token-based → FM
  - 在 Open X-Embodiment subset 上预训
- Ablation：
  - FM 步数：1, 4, 10, 50
  - Chunk size: 16, 32, 50
  - LoRA only fine-tune vs full fine-tune
- 评估：LIBERO + Bridge Data V2 上成功率
- 预期：性能与 OpenVLA 持平，但 inference 快 3-5×

**Experiment 3：真机验证（gap to deployment）**

- 目标：在国产机器人（如 BeiTong 7-DoF arm）上部署，测数据效率与失败模式
- Setup：
  - 收集 50 demos 真机数据
  - LoRA fine-tune from Experiment 2 model
  - 在 10 个 task 上 each 20 trials
- 评估：
  - Success rate
  - 失败模式分类（visual mis-perception / control timeout / grasp slip）
  - Data efficiency: 用 10/30/50/100 demos 训练，看 success rate 曲线
- 预期：50 demos 达到 70%+ 成功率（vs from-scratch 30%），证明 LoRA 路线有效

---

**Wrap-up**：
- 论文产出：Experiment 1+2 → ICRA/CoRL workshop（"FM-Action-Diffusion"）
- 论文产出：Experiment 1+2+3 → CoRL/ICRA 主会
- 后续：扩展到双手、长 horizon，做 RDT/Pi-0 替代品

---

## 评分标准

每题答出**关键技术点**得部分分。
- 实验设计题：**自我批判**与**对照实验**是高分关键
- 创新答案（已超出课程内容）：教学评语 + 满分

---

> **此 Quiz 完成 = 你已具备 VLA 研究者基础。下一步是动手做。**
