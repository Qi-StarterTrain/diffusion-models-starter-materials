# 论文导读 14: Rectified Flow (Liu et al., ICLR 2023)

**标题**：Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
**作者**：Xingchao Liu, Chengyue Gong, Qiang Liu (UT Austin)
**核心地位**：⭐⭐⭐⭐ Flow Matching 实践化的关键，"reflow" trick 源头

---

## 一、为什么必读

- 与 Lipman 的 Flow Matching 论文并行提出（同时期）
- 提出**reflow** trick：把训练好的 FM 进一步"拉直"
- 简洁实用，SD 3 的训练采纳了这些 idea

---

## 二、核心思想

### 2.1 Linear path

与 Flow Matching 同：
$$x_t = (1-t) x_0 + t x_1$$

训练 $v_\theta(x_t, t) \approx x_1 - x_0$。

### 2.2 Reflow 关键 trick

训完第一个模型 $v_\theta^{(1)}$ 后：
1. 生成大量 $(x_0, x_1^{\text{pred}})$ pairs（用 $v_\theta^{(1)}$ 采样）
2. 用 $x_1^{\text{pred}}$ 替代真实 $x_1$，重新训
3. 得到 $v_\theta^{(2)}$，trajectory 更直

### 2.3 为什么 reflow 有效？

问题：第一个 model 学到的"linear path"在实际采样时**不真的是直线**。因为多个 $x_1$ 共享同一 $x_0$，模型在中间产生 detour。

reflow 把 detour 直接"硬编码"成新数据，让 $v_\theta^{(2)}$ 学到真正的直线。

---

## 三、实验验证

### 3.1 Straightness 指标

定义 trajectory 的曲率：
$$S = \mathbb{E}\int_0^1 \|v(x_t, t) - (x_1 - x_0)\|^2 dt$$

低 $S$ = 直；高 $S$ = 弯。

实验：reflow 后 $S$ 显著下降。

### 3.2 1-step quality

| Model | 1-step FID |
|-------|-----------|
| FM (original) | 30+ |
| **FM + 1 reflow** | **5-10** |
| FM + 2 reflow | 接近 1 |

—— **1-step generation 几乎可行**（接近 Consistency Model）。

---

## 四、与 Consistency Model 的关系

- Consistency Model：通过 distillation 让 1-step 等价于多步
- Reflow：通过数据重生让 trajectory 变直
- 两者都能做 1-step，机制不同

**对比**：
- Reflow 不需要 EMA 或 teacher（自蒸馏）
- Consistency Model 不需要 reflow 数据

实践中两者可结合：reflow → distillation。

---

## 五、阅读建议

| 章节 | 优先级 |
|------|--------|
| §1 Intro | ✅ |
| **§3 Method** | 🔴 核心 |
| §3.2 Rectified Flow | 🔴 核心 |
| §3.3 Reflow procedure | 🔴 核心 |
| §4 Generation results | ✅ |
| §5 Translation | 选读（image2image） |

---

## 六、常被误读

### 1. "Reflow 必须做 distillation？"
**不**。Reflow 训出的 $v^{(2)}$ 可以**任意步数**采样，质量都好。distillation 是另一回事（CM 路线）。

### 2. "Rectified Flow ≠ Flow Matching？"
**几乎等价**，但侧重不同：
- Flow Matching (Lipman) 强调**框架推导**
- Rectified Flow (Liu) 强调**实践技巧 reflow**

工程上你看到 "linear path FM" = "rectified flow"。

---

## 七、思考题

1. 实现一个 toy 2D reflow 实验：对比 0/1/2/3 次 reflow 的 trajectory 曲率
2. 思考：reflow 是否会引入 "compounded error"（前一轮模型错误传到下轮）？
3. 比较 reflow 与 CM distillation 的训练成本

---

## 八、引用

```
@inproceedings{liu2023flow,
  title={Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
  author={Liu, Xingchao and Gong, Chengyue and Liu, Qiang},
  booktitle={ICLR},
  year={2023}
}
```
