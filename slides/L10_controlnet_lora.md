# L10: ControlNet 与 LoRA 进阶（W9）

> **目标**：区分空间条件控制与模型个性化，理解 ControlNet 的特征残差注入和 LoRA 的低秩权重更新
> **前置**：L08 (CFG), L09 (LDM)
> **核心问题**：当我们已经有 SD 这种大模型，怎么"低成本"加新能力？

---

## §1 课程定位

P0-P1 部分：从零造一个扩散模型。
P2 开始：在已有大模型上**做加法**。

常见的工程选择是复用预训练模型：

- 从零预训练代价高，优先保留已有能力；这不意味着主干不能微调
- 但用户需求多变（特定画风、特定姿态、特定 IP）
- 解决方案：**Plug-in 思路** → ControlNet、LoRA、Textual Inversion、IP-Adapter…

**核心三问**：

1. 加什么模块？（架构）
2. 训什么参数？（trainable subset）
3. 推理时怎么注入？（forward pass 修改）

**本讲范围**：§2–§4 以 SD 1.5 的 U-Net 为例。ControlNet 的完整副本包含数亿参数，并不是“小参数方法”；§5 也会介绍不属于严格 PEFT 的个性化方案。不同模型家族的模块名与接口不能直接混用。

---

## §2 ControlNet（Zhang et al., 2023）

### 2.1 动机

L08 在文本条件上使用了 CFG，但 **CFG 是条件与参考预测的组合方法，并不限于文本条件**。仅靠文字通常难以精确指定空间布局。例如我想：

- 生成一只猫，但**姿态**像我画的草图
- 生成一座建筑，但**深度结构**像参考图
- 生成一个人，但**关键点**像我提供的骨架

→ 需要**空间结构条件**。

---

### 2.2 架构核心

ControlNet 的关键 trick 三个：

**1. Trainable copy of U-Net encoder + middle block**

- 从原 SD U-Net 复制下采样编码路径和 middle block 的预训练权重；并非随机初始化整个副本
- 副本接受与主干相同的 noisy latent、时间步和文本条件
- condition image 经独立的条件编码网络降采样到 latent 的空间尺寸，再注入副本；它通常不经过图像 VAE

**2. Zero Convolution**

- 副本的多尺度特征通过 1×1 zero conv，得到与主干 skip features 及 middle feature 形状相同的残差
- 残差加到主干提供给上采样路径的 skip features，以及 middle block 的输出；**不是直接加到最终噪声预测**
- 这些连接卷积的权重和偏置**初始化为 0**

**3. Locked main UNet**

- 原 SD U-Net 权重冻结，训练副本、条件编码网络和 zero conv；VAE 与文本编码器通常也冻结

```
condition image → 条件编码网络 ─────────┐
                                      ↓
(z_t, t, text) → ControlNet 编码路径 + mid → zero conv → {r_skip}, r_mid

(z_t, t, text) → 冻结 U-Net 下采样路径 → {h_skip}, h_mid
                                          ↓
                         {h_skip + r_skip}, h_mid + r_mid
                                          ↓
                               冻结 U-Net 上采样路径
                                          ↓
                                       eps_pred
```

上图简写了多条连接。标准 SD 1.5 ControlNet 返回 12 个 down-path/skip 残差和 1 个 mid 残差；“down residual”表示它对应下采样路径保存的特征，随后被上采样路径消费。参考：[ControlNet 原论文 §3.2](https://arxiv.org/html/2302.05543v3)、[Diffusers ControlNet 输出接口](https://huggingface.co/docs/diffusers/api/models/controlnet)。

---

### 2.3 Zero Convolution 为什么有效

**核心问题**：把新增条件分支接到训好的 SD 上，怎样让初始预测保持不变？

**答案**：用 zero conv 让初始连接强度为 0。

```python
import torch.nn as nn

class ZeroConv(nn.Conv2d):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, kernel_size=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
```

**初始**：副本输出乘 0 = 没有影响，等价于原 SD。

**为什么全零连接还能学？** 对单个空间位置，将输出连接简写为 r=Wh+b，上游梯度为 g，则：

```math
\frac{\partial L}{\partial W}=gh^\top,\qquad
\frac{\partial L}{\partial b}=g,\qquad
\frac{\partial L}{\partial h}=W^\top g
```

初始 W=0 时，连接层仍可收到非零梯度，而经这条连接传给副本的梯度为 0。连接权重更新后，梯度才开始进入其上游；因此不会“因为全零初始化就永远学不动”。多层 zero conv 的具体梯度传播还取决于连接位置。

**开始学习不等于立刻学会控制**：有效结构控制可能在较多更新后才出现，不能承诺“训练几步就有效”。

> **与残差连接的关系**：ControlNet 用零初始化的附加残差保持初始输出不变。普通 ResNet 仅有 identity skip，并不保证随机初始化的残差分支输出为零。

---

### 2.4 训练

数据集：(condition image, target image, caption) 三元组。

以下展示 **SD 1.5 的 epsilon-prediction 训练步骨架**，省略数据加载与混合精度设置。预先冻结 U-Net / VAE / 文本编码器，仅让 ControlNet 可训练；optimizer 只管理 ControlNet 参数。text_emb 由冻结文本编码器获得，control_img 已按对应 checkpoint 的要求预处理，并与目标图像空间对齐。

标准 SD 1.5 示例中，text_emb 为 (B, 77, 768)，control_img 为 (B, 3, 512, 512)，条件图通常使用 [0, 1] 范围；这与 VAE 目标图的 [-1, 1] 不同。设备、dtype 和实际预处理应与所用 checkpoint 一致。

```python
import torch
import torch.nn.functional as F

optimizer.zero_grad(set_to_none=True)
with torch.no_grad():
    # target_img: (B, 3, 512, 512)，范围 [-1, 1]
    z_0 = vae.encode(target_img).latent_dist.sample()
    z_0 = z_0 * vae.config.scaling_factor

true_eps = torch.randn_like(z_0)
t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                  (z_0.shape[0],), device=z_0.device).long()
z_t = noise_scheduler.add_noise(z_0, true_eps, t)

down_res, mid_res = controlnet(
    z_t, t, encoder_hidden_states=text_emb,
    controlnet_cond=control_img, return_dict=False,
)
eps_pred = unet(
    z_t, t, encoder_hidden_states=text_emb,
    down_block_additional_residuals=down_res,
    mid_block_additional_residual=mid_res,
).sample
loss = F.mse_loss(eps_pred.float(), true_eps.float())
loss.backward()
optimizer.step()
```

**冻结权重 ≠ 禁止梯度经过模块**。U-Net 参数设 `requires_grad_(False)`，但上面的整个 U-Net 调用不能包在 `torch.no_grad()` 中，否则梯度无法从 loss 经冻结的上采样路径传回 ControlNet 残差。VAE 编码和冻结文本编码可以在 `no_grad` 下执行。

ControlNet 的训练目标仍是扩散去噪损失，模型学的是“在给定文本与空间条件下预测噪声”。其他预测参数化需要相应修改 target，不能一律使用 true_eps。参考：[Diffusers 官方 ControlNet 训练脚本](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py)。

---

### 2.5 推理（CFG + ControlNet）

```python
# 采样循环中的一步，普通模式 guess_mode=False；输入已在同一设备且 dtype 匹配
# text_emb = cat([参考文本 embedding, 正提示词 embedding])，形状 (2B, 77, 768)
with torch.no_grad():
    z_in = torch.cat([z, z], dim=0)
    z_in = scheduler.scale_model_input(z_in, t)
    control_in = torch.cat([control_img, control_img], dim=0)
    down_res, mid_res = controlnet(
        z_in, t, encoder_hidden_states=text_emb,
        controlnet_cond=control_in, conditioning_scale=control_strength,
        guess_mode=False, return_dict=False,
    )
    eps_pair = unet(
        z_in, t, encoder_hidden_states=text_emb,
        down_block_additional_residuals=down_res,
        mid_block_additional_residual=mid_res,
    ).sample
    eps_ref, eps_cond = eps_pair.chunk(2)
    eps = eps_ref + cfg_scale * (eps_cond - eps_ref)
    z = scheduler.step(eps, t, z).prev_sample
```

注意：

- `control_strength` 缩放的是**注入特征的残差**，不是对最终图像或噪声预测做线性混合；0 使该分支不影响主干预测，但若仍执行分支并不节省计算。
- 普通模式中，两条 CFG 分支都接收相同的空间条件；参考文本为空或为 negative prompt，所以“unconditional”不等于没有任何条件。`guess_mode` 的条件分支计算方式另有约定，不要和这段混用。
- CFG scale 放大文本条件与参考预测的差异，control strength 调整结构残差；两者会相互影响，不能当成完全独立的旋钮。
- 多个 ControlNet 通常将各自加权的同形状残差逐层求和，再交给 U-Net；条件冲突不会自动得到正确解决，需要调整各自权重、启用时间区间或输入条件。

参考：[Diffusers ControlNet 推理实现](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet.py)。

---

### 2.6 常见 condition 类型

| 类型 | 提取工具 | 用途 |
|------|---------|------|
| Canny edge | OpenCV cv2.Canny | 保持轮廓 |
| Depth | MiDaS / ZoeDepth | 保持 3D 结构 |
| Pose (OpenPose) | OpenPose detector | 保持人物姿态 |
| Semantic segmentation | 与 checkpoint 匹配的语义分割器及类别调色板 | 保持语义区域布局 |
| Normal map | 法线估计网络 | 保持表面方向 |
| Scribble | 手画 | 用户草图 |

**预处理属于模型接口**：边缘的黑白方向、深度归一化、骨架绘制格式、语义分割的类别颜色都应匹配训练约定。SAM 的实例掩码不自动等同于语义分割图，不能不经转换就替代。选用已针对该条件训练的 ControlNet，不是把任意条件图送入任意权重都有效。

---

## §3 LoRA（Hu et al., 2021）

### 3.1 动机

完整 ControlNet 新增数亿参数，主要解决空间条件接入。如果需求是**保留我家猫的身份特征**，或学习某种画风，则可以用 LoRA 做参数高效微调；主体身份与画风是两类任务，不应混为一谈。

LoRA 只训练选定层的低秩增量，不复制整个 U-Net。文件大小取决于目标层数、rank、精度及是否训练文本编码器，并非固定几 MB。

---

### 3.2 数学核心

对任意线性层 $`W \in \mathbb{R}^{d \times k}`$，LoRA 把更新参数化为低秩矩阵：

```math
W_{\text{new}} = W + \Delta W = W + \frac{\alpha}{r}BA
```

其中 $`B \in \mathbb{R}^{d \times r}`$，$`A \in \mathbb{R}^{r \times k}`$，通常选择 $`r\ll\min(d,k)`$，因此更新矩阵的秩至多为 r。标准 LoRA 的缩放为 $`\alpha/r`$；alpha 是超参数，不能从公式中省略，除非明确设 alpha=r。其他 LoRA 变体可能采用不同缩放，本讲固定讨论标准形式。

**该层可训练参数量**从 $`dk`$ 降到 $`r(d+k)`$。例如 d=k=1024、r=8：1,048,576 → 16,384，恰好减少 **64 倍**。基础权重 W 仍需保存和执行，不能把这个比值当作训练耗时或总显存的下降倍数。

参考：[LoRA 原论文 §4.1](https://arxiv.org/html/2106.09685v2)。

---

### 3.3 训练初始化

```math
A \sim \mathcal{N}(0, \sigma^2), \quad B = 0
```

所以 $`\Delta W=(\alpha/r)BA=0`$，初始等价于原模型（与 zero conv 保持初始输出的思路类似）。

训练时：原 $`W`$ 冻结，只训 $`A, B`$。

初始 B=0 时，A 的梯度为 0，B 通常能先获得非零梯度；B 更新后 A 也能开始学习。**不要把 A、B 同时初始化为 0**，否则标准 BA 参数化下两者的梯度都会为 0。

---

### 3.4 推理时合并

```math
W_{\text{final}} = W + \lambda\frac{\alpha}{r}BA
```

其中 lambda 是推理时的 adapter 强度。对可合并的标准线性层，提前融合后仍然只执行一次原尺寸矩阵乘法，因此没有额外 LoRA 分支计算；**融合操作本身有一次性开销**。保留为 adapter 时：
```math
y = Wx + \lambda\frac{\alpha}{r}B(Ax)
```

它便于动态切换强度，但要额外计算低秩分支。多个兼容同一主干的 LoRA 可以组合：

```math
W_{\text{mix}}=W+\sum_i\lambda_i\frac{\alpha_i}{r_i}B_iA_i
```

这是权重增量相加，不保证语义效果线性叠加或互不干扰。alpha/r 是训练配置的一部分，lambda 是运行时强度；它们也不同于 L09 的 VAE latent scaling factor。

---

### 3.5 在 SD 中的应用

SD UNet 的哪些层加 LoRA？

- attention 的 Q、K、V、output 投影是常见起点，可同时覆盖 self-attention 和 cross-attention。
- FFN、卷积或文本编码器也可以适配，取决于实现与任务；不存在普遍的“cross-attention 最好、FFN 不值得训练”。
- `target_modules` 按模块名匹配，不是按语义自动选择；SD 1.5 的 `to_q` 等后缀通常会同时命中 self-attention 与 cross-attention。

示例配置（假设 unet 已加载，使用 Diffusers + PEFT）：
```python
from peft import LoraConfig

unet.requires_grad_(False)
rank = 8  # 可对 4 / 8 / 16 / 32 做验证集扫描
config = LoraConfig(
    r=rank,
    lora_alpha=rank,  # 标准缩放 alpha/r = 1
    init_lora_weights="gaussian",  # A 随机、B 为 0
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
)
unet.add_adapter(config)
trainable_params = [p for p in unet.parameters() if p.requires_grad]
print("trainable parameters:", sum(p.numel() for p in trainable_params))
# optimizer 只接收 trainable_params
```

如果只想训练 cross-attention，需要进一步按完整模块路径筛选（SD 1.5 常见 `attn2`），并打印实际命中的层验证。参考：[Diffusers LoRA 训练指南](https://huggingface.co/docs/diffusers/training/lora)、[PEFT LoRA 配置](https://huggingface.co/docs/peft/package_reference/lora)。

---

### 3.6 LoRA 训练规模

小样本主体定制和画风适配对数据覆盖的要求不同；张数、步数、显存没有统一答案。固定数据集后，至少报告：图像数、分辨率、目标层、rank、alpha、有效 batch、学习率、优化步数、是否训练文本编码器及精度。

例如，10 张图、有效 batch=1、500 个优化步，在每步消费一个样本的简化设置下，相当于每张图平均被看到 50 次；batch=4 时则约为 200 次。**只看“500 步”无法判断是否训练充分或过拟合**。

LoRA 显著减少梯度、优化器状态和可发布增量的大小，但仍需通过主干做前向与必要的反向传播。具体加速与显存节省应在同一硬件和训练配置下测量，不能承诺比全量微调便宜两个数量级。

---

## §4 ControlNet vs LoRA：本质对比

| 维度 | ControlNet | LoRA |
|------|-----------|------|
| 目标 | 增加新**条件**通道 | 微调现有**权重** |
| 改变什么 | 在 U-Net 内部注入多尺度残差 | 选定层的有效权重增加 ΔW |
| 训练参数 | 完整 SD 1.5 版本为数亿级 | 各目标层的 r(d+k) 之和，取决于配置 |
| 数据要求 | 对齐的条件图、目标图及可选文本 | 任务相关图文数据；主体定制可采用小样本 |
| 用途 | 边缘、深度、姿态等空间控制 | 主体、画风或其他任务适配 |
| 推理开销 | 每步增加条件分支前向，具体比例需实测 | 标准线性层融合后无额外分支；未融合有额外计算 |

**组合使用**：可将 ControlNet 的空间条件、LoRA 的参数适配与 CFG 结合，但不是所有任务都需要全开。需要检查底模兼容性、条件冲突和各自强度；LoRA 也能用于适配某些 ControlNet 层，两者并非互斥类别。

---

## §5 其他个性化与条件适配方法（概览）

这些方法操作的位置不同，并不都属于严格的 PEFT（参数高效微调）。尤其 **DreamBooth 是主体定制的训练方案，LoRA 是参数更新方式，二者可以组合**。

### 5.1 Textual Inversion (Gal et al., 2022)

冻结生成模型及文本编码器的其他参数，只学习代表新概念的 token embedding。SD 1.5 的单个向量维度为 768；若使用 m 个学习向量，参数量为 m×768，换文本编码器后维度也会变化。

推理时在 prompt 中使用对应的占位符，例如 `<concept>`，将学习到的表示送入文本编码器。它能表达主体或画风，但不提供 ControlNet 那样的显式空间布局输入；不要理解成“完全学不到结构信息”。参考：[Textual Inversion 原论文](https://arxiv.org/abs/2208.01618)。

### 5.2 DreamBooth (Ruiz et al., 2022)

用少量主体图像，把一个标识符与特定主体绑定，并可加入 **class-specific prior preservation loss**，约束模型继续表达同类其他实例，缓解语言漂移与过拟合；它不保证完全消除遗忘。

原始 DreamBooth 是生成模型微调方法，不专属于 SD U-Net，也不等同于一种 adapter 架构。在 SD 实践中可以全量微调 U-Net、按配置微调文本编码器，也可以只更新 LoRA 参数，即 **DreamBooth-LoRA**。质量与显存取决于实现和数据，不能统一写为“必需 10+ GB”。参考：[DreamBooth 原论文](https://arxiv.org/abs/2208.12242)、[Diffusers DreamBooth 训练说明](https://huggingface.co/docs/diffusers/training/dreambooth)。

### 5.3 IP-Adapter (Ye et al., 2023)

把参考图像作为 prompt：冻结的图像编码器提取特征，学习到的投影模块将其变为图像条件，并通过**与文本分开的 cross-attention 分支**注入生成模型；主干与图像编码器保持冻结，训练新增的图像条件模块。

它常用于参考外观、画风或图像变体，可与 ControlNet 的空间条件互补。但“IP 决定 what、ControlNet 决定 where”只是粗略类比：两类条件都会影响内容与布局，也可能冲突。参考：[IP-Adapter 原论文 §3.2](https://arxiv.org/html/2308.06721v1)。

### 5.4 Custom Diffusion (Kumari et al., 2023)

主要更新文本 cross-attention 中的 K、V 投影，并学习新概念的 token 表示；它是选择性更新已有矩阵，不等同于把这些矩阵参数化成 LoRA。

该方法还研究了多概念联合训练与模型组合。训练速度和组合质量需结合论文中的数据、硬件与基线理解，不能概括为“固定快 6×”或“多概念一定优于 LoRA”。参考：[Custom Diffusion 原论文](https://arxiv.org/abs/2212.04488)、[Diffusers Custom Diffusion 训练说明](https://huggingface.co/docs/diffusers/training/custom_diffusion)。

---

## §6 工程实战要点

### 6.1 ControlNet 训练

- **先保证空间对齐**：目标图与条件图的 resize、crop、flip 必须同步；条件图的格式与归一化匹配任务。条件本来就应约束目标结构，关键是避免条件错配或无意把完整目标 RGB 当作“条件”泄漏进去。
- **按任务处理文本**：caption 描述内容，结构图指定布局；可用一定比例的空文本训练来降低对 caption 的依赖，比例通过验证决定，不要求条件与目标“不能完全对应”。
- **有效 batch 与单卡 batch 分开报告**：有效 batch = 单卡 micro-batch × GPU 数 × 梯度累积步数。不要求 batch 至少 128；从硬件可承受的 micro-batch 开始，按需累积。
- **按验证结果确定步数**：查看条件遵循度、图像质量及未见布局上的泛化，而不是统一训练 50K 步。与随机初始化相比，复制主干权重提供了更好的起点，但不保证少量数据一定足够。

可参考 [Diffusers 官方 ControlNet 训练脚本](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py) 配置数据、梯度累积与验证。

### 6.2 LoRA 训练

- **rank 是容量预算，不是任务复杂度的固定映射**：固定目标层和 alpha/r，再比较 r=4 / 8 / 16 / 32。较大 rank 不必然过拟合，较小 rank 也可能记住训练图。
- **学习率是配置相关的起点**：SD 1.5 U-Net attention LoRA 可以从 1e-4 量级试起；文本编码器、不同优化器或底模可能需要更低学习率。1e-5 不是错误值，也不存在统一的“SD 训练学习率”。
- **用有效样本曝光量理解步数**：同时记录数据重复次数、有效 batch 和优化步数；保留固定验证 prompt / seed，比较主体相似度、画风、可编辑性与背景多样性。
- **检查是否真的省在预期位置**：基础权重冻结，optimizer 仅接收目标可训练参数；必要时使用低精度、梯度检查点或优化器状态压缩。保存多个 checkpoint，按验证表现选择，而非只看训练 loss。

### 6.3 LoRA 调试

- **Loss 不降或输出无变化**：检查实际命中的层、`requires_grad`、optimizer 参数列表、学习率与 target；初始 B=0 时 A 的首步梯度为 0 是正常现象。
- **复现训练图背景或姿态、换场景失败**：检查数据多样性、caption 是否把背景与主体混淆、训练曝光量和推理强度；再比较 rank，而不是直接认定“rank 太大”。
- **影响不含主体词的 prompt**：普通 LoRA 加载且强度非零时，会改变匹配层在所有 prompt 下的有效权重。trigger word 只是学习到的文本线索，**不是启用开关**。要恢复原模型行为，应禁用 / 卸载 adapter，或将其强度设为 0；已经融合进权重时还需撤销融合或恢复底模。

---

## §7 与 VLA / 具身的连接

“复用预训练能力，再适配动作任务”的思路在**具身智能**领域也有应用，但不能把所有动作模块都称为 ControlNet：

- **LoRA 的直接迁移**：OpenVLA 官方实现提供了 LoRA 微调路径，用机器人示范适配预训练策略。它复用了低秩权重更新机制，不需要把动作任务改写成图像 ControlNet。[OpenVLA 官方仓库](https://github.com/openvla/openvla)
- **RT-2**：将机器人动作表示为 token，并通过视觉语言数据与机器人数据共同微调 VLM，不能简化为“主干冻结 + action token adapter”。[RT-2 原论文](https://arxiv.org/abs/2307.15818)
- **π₀**：在预训练 VLM 基础上结合动作 expert，以 flow matching 建模连续动作序列；与 RT-2 的离散 token 路线不同，也不等同于 ControlNet 的零卷积空间残差。[π₀ 原论文](https://arxiv.org/abs/2410.24164)

可迁移的是“明确哪些能力复用、哪些参数适配、哪些条件进入模型”的分析方式。是否冻结主干、是否零初始化、是否保持初始输出等价，要逐个架构判断，不能由“使用了预训练模型”直接推出。

---

## §8 课后任务

### 必做

1. 在 Project 3 作业库提交 `04_controlnet_demo.ipynb`：跑通 SD 1.5 + Canny ControlNet，打印 down / mid 残差形状；在相同初始 latent、prompt 和 scheduler 下比较 control strength=0 与无 ControlNet 的预测，验证其数值接近。
2. 用兼容 SD 1.5 的 Canny + Pose 两个 ControlNet，对齐条件图后组合推理。固定 seed，对比单条件、相容双条件和冲突双条件，扫描控制权重并解释结果；不要假定模型会自动解决冲突。
3. 训一个 LoRA：可从 10 张主体图、rank=8、500 个优化步开始，记录 alpha、目标层、有效 batch、学习率及实际训练时间。对比底模、训练中间 checkpoint 和最终 checkpoint 在固定验证 prompt 下的效果，并比较有 / 无 trigger word 与 adapter strength=0 / 非零的差异。这里的 500 步是实验起点，不是质量保证。

### 进阶

4. 阅读 IP-Adapter 论文，理解它与 ControlNet 的设计差异
5. 实现 LoRA + ControlNet 组合推理（注意训练数据来源不同）

---

## §9 FAQ

**Q1：为什么 ControlNet 的副本不直接训练新 UNet？**

A：副本复用预训练下采样路径与 mid 权重，新增分支学习空间控制，冻结主干保留已有生成能力。它接收同一 noisy latent，并输出形状匹配的多尺度残差，但训练后副本特征不必与主干完全相同。是否从零训练新 U-Net 是成本、数据与任务的选择，并非数学上不能做。

---

**Q2：LoRA rank 怎么选？**

A：对本讲的 SD 1.5 attention LoRA，可从 r=8 开始，在同一数据与目标层上比较 r=4 / 8 / 16 / 32，并保持 alpha/r 的约定一致。rank 增大会增加容量和可训练参数，但是否过拟合由数据、曝光量、正则化等共同决定；LoRA 对输出的影响还取决于学到的增量和推理强度，不能只看 rank 判断。

---

**Q3：能不能"训一次 LoRA，所有 SD 版本通用"？**

A：**不行**。LoRA 与 base model 绑定：

- SD 1.5 的 LoRA 不能直接用在 SDXL（维度都不同）
- 同一 SD 1.5 的不同 fine-tune（如 DreamShaper）上 LoRA 效果会变
- 即使维度匹配，还需检查模块命名、底模版本、文本编码器和缩放约定；“能加载”不等于“效果保持一致”

---

**Q4：为什么主干冻结了，ControlNet / LoRA 训练仍然需要显存和反向传播？**

A：冻结只是不更新主干参数；训练新增分支仍需计算损失对这些分支的梯度。梯度可能经过冻结模块，中间激活也可能需要保存。因此可训练参数少不等于整个训练计算量同比下降，不能把训练中的主干整体放进 `no_grad`。见 §2.4 与 §3.6。

---

**Q5：trigger word 能让 LoRA 只在指定 prompt 下生效吗？**

A：普通 LoRA 不能。trigger word 帮模型将文本与学习概念关联；真正决定 adapter 是否参与前向的是加载状态与运行时强度。若需要按 prompt 自动启停，应由应用逻辑显式切换 adapter，而不是依赖 trigger word 保证其他 prompt 完全不受影响。

---

## §10 参考资源

- [Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models*](https://arxiv.org/abs/2302.05543), ICCV 2023（ControlNet）
- [Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685), ICLR 2022
- [Gal et al., *An Image is Worth One Word*](https://arxiv.org/abs/2208.01618), 2022（Textual Inversion）
- [Ruiz et al., *DreamBooth*](https://arxiv.org/abs/2208.12242), CVPR 2023
- [Ye et al., *IP-Adapter*](https://arxiv.org/abs/2308.06721), 2023
- [Kumari et al., *Multi-Concept Customization of Text-to-Image Diffusion*](https://arxiv.org/abs/2212.04488), CVPR 2023（Custom Diffusion）
- [Diffusers ControlNet 训练脚本](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py)、[LoRA 训练说明](https://huggingface.co/docs/diffusers/training/lora)
- [Hugging Face PEFT LoRA 文档](https://huggingface.co/docs/peft/package_reference/lora)

---

> 下一讲（L11）我们离开 SD/UNet 时代，转向**Diffusion Transformer**——Sora 与 SD 3 的架构底座。
