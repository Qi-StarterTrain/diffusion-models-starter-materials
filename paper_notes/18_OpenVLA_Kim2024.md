# 论文导读 18: OpenVLA (Kim et al., 2024)

**标题**：OpenVLA: An Open-Source Vision-Language-Action Model
**作者**：Moo Jin Kim et al. (Stanford, UC Berkeley, Google DeepMind)
**核心地位**：⭐⭐⭐⭐⭐ 开源 VLA 标杆，工业研究起点

---

## 一、为什么必读

- **目前最强开源 VLA**（截至 2024 末）
- 全开放：weights + code + training data + recipe
- 击败 RT-2-X（Google 闭源），且尺寸更小
- 国内复现 + LoRA fine-tune 的事实标准

---

## 二、阅读路线图

| 章节 | 优先级 |
|------|--------|
| §1 Intro | ✅ |
| §2 Related Work | 跳过 |
| **§3 Method** | 🔴 核心 |
| §3.1 Model Architecture | 🔴 核心 |
| §3.2 Action Tokenization | 🔴 核心 |
| §3.3 Training | 🔴 核心 |
| §4 Experiments | 🔴 必看 |
| §5 LoRA Fine-tuning | 🔴 必看 |

---

## 三、模型架构

```
Image (224×224 × 2 cameras)
   ↓ DINOv2 + SigLIP (双视觉 encoder)
   ↓ concatenate features
   ↓ MLP projection
Visual tokens (~256)
   ↓
   ┌─ + Language tokens ─┐
   ↓                      ↓
Llama 2 7B (decoder only)
   ↓
Action tokens (7 dims × 256 bins = 7 tokens)
   ↓ De-tokenize
End-effector delta pose (xyz + rpy + gripper)
```

**关键设计**：
- **双视觉 encoder**（DINOv2 + SigLIP）：互补的视觉特征
- **Llama 2 7B 作为 backbone**：复用 LLM 预训练
- **Action 作为离散 token**：与 text 同等级处理

---

## 四、Action Tokenization

把 7 维 continuous action 离散化：
- 每维 256 bins（按 quantile 划分）
- 共 7 个 tokens 表示一个 action

**为什么离散**：让 action 直接进入 LLM 的 token vocabulary，无需 separate head。

—— 与 RT-2 同思路。

---

## 五、训练

### 5.1 数据

**Open X-Embodiment**：跨 22 个机器人 embodiment，970K episodes。
- 数据格式统一化（不同 robot 都映射到 7-DoF end-effector）
- 970K episodes × ~100 frames ≈ 100M 帧

### 5.2 训练 recipe

- 64 个 A100，14 天
- Batch 256
- Constant LR 2e-5
- 不冻结任何 module（全 fine-tune）

---

## 六、性能（与 RT-2-X 比较）

LIBERO benchmark（4 个 task suites）：

| Model | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long |
|-------|----------------|---------------|-------------|-------------|
| RT-2-X | 0.0 | 0.0 | 0.0 | 0.0 |
| Octo-Base | 78.9 | 85.7 | 84.6 | 51.1 |
| **OpenVLA (zero-shot)** | **84.7** | **88.4** | **79.2** | **53.7** |

→ Zero-shot 击败 RT-2-X by huge margin（RT-2-X 几乎 0 是因为 LIBERO 与训练数据 distribution mismatch，OpenVLA 因更广泛训练数据更 robust）。

---

## 七、LoRA Fine-tune（§5 必看）

OpenVLA 的杀手锏：**用少量数据 + LoRA 适配新 robot**。

### 7.1 设置
- Target modules：`q_proj, k_proj, v_proj, o_proj` of Llama
- Rank 32
- Adapter ~110M params (vs 7B base)

### 7.2 结果

50 demonstrations 即可在 BridgeData V2 新 task 上达到 80% 成功率。

→ **VLA 的实用化路径**：base VLA 一次训练，每 robot/task 用 LoRA 适配。

---

## 八、Inference 速度

OpenVLA 在 RTX 4090 上：
- ~3 Hz（无 quantization）
- ~10 Hz（int8 quantization）
- 仍不足以做 high-frequency control（如 50 Hz）

→ 需要 distillation（CM 路线）或更小模型（Pi-0 用更小 base）才能实时。

---

## 九、常被误读

### 1. "OpenVLA 比 RT-2 强？"
**Zero-shot 是的，但 RT-2 闭源 fine-tune 后未知**。OpenVLA 的优势在**开源 + 同等性能**。

### 2. "OpenVLA = LLM with action head?"
**几乎是**。具体：Llama 2 (frozen pre-train) + visual encoder + 微调全部参数 to predict action tokens.

### 3. "OpenVLA 能 zero-shot generalization 到任意 robot?"
**否**。Zero-shot 限于训练集见过的 embodiment / camera viewpoint。新 robot 还是要 LoRA fine-tune。

---

## 十、与 Pi-0 / RDT 对比

| | OpenVLA | Pi-0 | RDT-1B |
|---|---------|------|--------|
| Params | 7B | 3.5B | 1B |
| Action head | Discrete token | FM | Diffusion |
| Pretrained backbone | Llama 2 | PaliGemma | T5 + DiT |
| Open source | ✓ all | partial | ✓ all |
| Action freq | 3-10 Hz | 50-100 Hz | 30 Hz |
| Best at | Long-horizon language | Real-time control | Bimanual |

OpenVLA 强在 **language understanding**，Pi-0 强在 **real-time**，RDT 强在 **双手**。

---

## 十一、思考题

1. 为什么 OpenVLA 用 token-based action 而 Pi-0 用 FM？哪种更适合什么场景？
2. LoRA 在 VLA 上的 rank 选择是否与 LLM 不同？做实验
3. OpenVLA + LCM-LoRA 风格的 distillation 可行吗？设计方案
4. 思考：把 OpenVLA backbone 换成 Qwen2-VL（中文 VLM），数据用 AgiBot World，能做出"中文 VLA"吗？

---

## 十二、引用

```
@article{kim2024openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={Kim, Moo Jin and others},
  year={2024}
}
```

GitHub: https://github.com/openvla/openvla
