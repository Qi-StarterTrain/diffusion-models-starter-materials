# Experiment Log Template

> **使用原则**：**所有实验**（包括失败的）都必须记录。失败的记录比成功的更有价值。
> **命名规范**：`exp_YYYYMMDD_NN_short_description.md`（NN 是当天第几个实验）
> **存放位置**：项目仓库的 `logs/` 目录下。
> **提交频率**：每次跑实验前先开日志，跑完立即填写。

---

## Experiment ID

`exp_20250115_03_ddpm_cosine_schedule`

## 一句话目标

> 本次实验最想验证什么？答案应当是 yes/no 形式可判定的。

*示例：cosine schedule 在 200 epoch 内是否比 linear schedule 收敛更快？*

---

## 1. 环境信息

| 项目 | 内容 |
|------|------|
| 实验日期 | 2025-01-15 |
| 实验者 | Qugank |
| 硬件 | RTX 3090 24GB × 1 |
| 框架版本 | PyTorch 2.1.0, CUDA 12.1 |
| 仓库 | github.com/xxx/diffusion_course |
| Git commit | `a3f7c2e`（**必填**，便于复现） |
| Git 分支 | `feat/cosine-schedule` |

---

## 2. 假设与预期

> 在跑实验**之前**写下你的预期。之后对比结果，培养科学直觉。

**假设**：
- cosine schedule 在低 t 区域破坏图像更平缓，应当收敛更快

**预期结果**：
- 100 epoch 时，cosine 的 FID 比 linear 低 1–3 个点
- 训练 loss 曲线 cosine 更平滑

**如果与预期不符可能的原因**：
- 数据集太小，差异被噪声淹没
- 超参未调优，linear 处于欠拟合

---

## 3. 配置（完整 hyperparameters）

```yaml
# 必须保存为 yaml/json 文件，并附在仓库中
dataset: cifar10
image_size: 32
batch_size: 128
num_epochs: 200

model:
  type: unet
  base_channels: 128
  channel_mult: [1, 2, 2, 2]
  attention_resolutions: [16]
  num_res_blocks: 2

diffusion:
  T: 1000
  beta_schedule: cosine   # 本次实验变量
  beta_start: 0.0001
  beta_end: 0.02

optimizer:
  type: adamw
  lr: 2.0e-4
  weight_decay: 0.0

ema_decay: 0.9999
mixed_precision: fp16
seed: 42
```

---

## 4. 运行命令（必须可直接复制粘贴运行）

```bash
python train.py \
  --config configs/cifar10_cosine.yaml \
  --output_dir runs/exp_20250115_03 \
  --seed 42 \
  --resume false
```

---

## 5. 结果

### 5.1 数值指标

| 指标 | 数值 | 备注 |
|------|------|------|
| 训练总时长 | 8.5 h | |
| 显存峰值 | 18.2 GB | |
| 最终训练 loss | 0.024 | |
| FID @ 100 epoch | 8.4 | 5000 张样本 |
| FID @ 200 epoch | 5.7 | 5000 张样本 |
| IS @ 200 epoch | 8.9 ± 0.1 | |

### 5.2 可视化结果

> 至少附 3 张：训练 loss 曲线、生成样本网格、对比图。
> 文件应保存在 `runs/exp_xxx/` 下。

- `runs/exp_20250115_03/loss_curve.png`
- `runs/exp_20250115_03/samples_epoch_200.png`
- `runs/exp_20250115_03/comparison_with_baseline.png`

### 5.3 与 baseline 对比

| 配置 | FID | 训练时长 | 备注 |
|------|-----|----------|------|
| linear schedule (baseline) | 7.2 | 8.3 h | exp_20250112_01 |
| cosine schedule (本次) | 5.7 | 8.5 h | |

---

## 6. 结论

### 6.1 假设是否成立

✅ 假设成立 / ❌ 假设不成立 / 🟡 部分成立

**简短说明**：cosine 在 200 epoch 时确实优于 linear（FID 5.7 vs 7.2），但在 50 epoch 内两者差异不显著，与"收敛更快"的预期不完全吻合。

### 6.2 现象记录

观察到但与预期无关的现象：
- 训练前 10 个 epoch，cosine 的 loss 比 linear 高
- 生成样本中，linear 在低分辨率细节（边缘）更清晰，cosine 在整体颜色分布更自然

---

## 7. 失败记录与踩坑（**强制填写**）

> 没有遇到任何问题是不正常的。仔细回想是否有任何"花了 5 分钟以上才解决的事情"。

### 坑 1：fp16 训练前 100 步出现 NaN

- **现象**：开启 fp16 后，loss 在第 80 步左右变成 NaN
- **猜测原因**：U-Net 中 GroupNorm 的方差计算下溢
- **验证过程**：将 GroupNorm 改回 fp32 计算，问题消失
- **最终方法**：在 GroupNorm 层强制用 fp32（`autocast` 包裹），其他层 fp16
- **耗时**：约 2 小时
- **教训**：fp16 不是无脑开就能用，要注意 normalization 层的数值稳定性

### 坑 2：（如有，按上述格式记录）

---

## 8. 下一步计划

> 基于本次结果，下一个实验应该做什么？写得越具体越好。

- [ ] 在 LSUN-Bedrooms（更大数据集）上验证 cosine 优势是否仍然存在
- [ ] 试验 sigmoid schedule，看是否进一步提升
- [ ] 排查 cosine 在前 10 epoch loss 偏高的原因（是否与 SNR 分布有关）

---

## 9. 给未来自己的提醒

> 一句话总结这次实验最值得记住的事情。

*示例：跑长实验前一定要先在小数据集上做 50 epoch 的 dry-run，确认 fp16 不出 NaN。*

---

## 10. 提交清单（自检）

- [ ] 配置文件已保存到 `configs/`
- [ ] Git commit hash 已记录（且 commit 已 push）
- [ ] 训练曲线图已保存
- [ ] 至少 1 张样本网格图已保存
- [ ] WandB run 链接已附上（如使用）
- [ ] 失败记录至少 1 条
- [ ] 与 baseline 对比表已填写
