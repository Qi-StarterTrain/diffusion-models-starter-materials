# PyTorch 速查 · 扩散模型必备操作

> 本材料覆盖训练扩散模型最常用的 PyTorch 操作。
> 建议在 Jupyter 中实际跑一遍每段代码。

---

## §1 张量基础操作

### 1.1 创建张量

```python
import torch

# 从数据创建
x = torch.tensor([1, 2, 3])
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 特定形状
zeros = torch.zeros(3, 4)
ones = torch.ones(3, 4)
randn = torch.randn(3, 4)         # 标准正态分布
rand = torch.rand(3, 4)           # 均匀分布 [0, 1]
arange = torch.arange(0, 10)      # [0, 1, 2, ..., 9]
linspace = torch.linspace(0, 1, 5)  # 5 个均匀点

# 与已有张量同形状
zeros_like = torch.zeros_like(x)
randn_like = torch.randn_like(x)  # ⭐ 扩散模型加噪用得最多
```

---

### 1.2 形状操作（**必练**）

```python
x = torch.arange(24).reshape(2, 3, 4)
# x.shape == (2, 3, 4)

# reshape / view
x.view(6, 4)        # 重塑（要求 contiguous）
x.reshape(6, 4)     # 自动处理 contiguous（更安全）
x.flatten()         # (24,)
x.flatten(start_dim=1)  # (2, 12)

# permute / transpose
x.transpose(0, 1)   # 交换两个维度
x.permute(2, 0, 1)  # 任意排列维度，新形状 (4, 2, 3)
# ⚠️ permute 后通常不 contiguous，下一步是 view 时要 .contiguous()

# squeeze / unsqueeze
y = torch.zeros(1, 3, 1, 4)
y.squeeze()         # 去掉所有为 1 的维度 → (3, 4)
y.squeeze(0)        # 去掉指定维度（仅当为 1 时） → (3, 1, 4)
x.unsqueeze(0)      # 在 0 位置插入维度 → (1, 2, 3, 4)
```

---

### 1.3 广播规则

PyTorch 广播规则：
1. 从最右维度开始对齐
2. 每个维度要么相等，要么有一个为 1
3. 不存在的维度视为 1

```python
a = torch.zeros(3, 1, 4)
b = torch.zeros(2, 4)
# 对齐后：a (3, 1, 4)，b (1, 2, 4)
# 广播结果：(3, 2, 4)
(a + b).shape  # torch.Size([3, 2, 4])
```

> **🔑 在扩散模型中**：把 timestep embedding (B, dim) 加到 feature map (B, C, H, W) 时需要 `unsqueeze` 后广播，常见模式：
> ```python
> emb = emb.view(B, dim, 1, 1)  # 然后可以与 (B, C, H, W) 相加
> ```

---

## §2 张量索引与切片

```python
x = torch.randn(8, 3, 32, 32)  # batch of CIFAR images

x[0]           # 第一张图 (3, 32, 32)
x[:4]          # 前 4 张
x[:, 0]        # 所有图的 R 通道 (8, 32, 32)
x[..., :16, :16]  # 所有图的左上角 16x16 区域

# 高级索引（gather）
indices = torch.tensor([0, 2, 5])
x[indices]     # 选择第 0, 2, 5 张图

# 布尔索引
mask = x > 0
x[mask].shape  # 一维张量，包含所有正值
```

---

## §3 计算图与梯度

### 3.1 自动微分基础

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 2*2 + 3 = 7
```

---

### 3.2 计算图的"枝节"

```python
# detach: 切断梯度（前向值不变，但反向传播在此停止）
x = torch.randn(3, requires_grad=True)
y = x.detach()  # y 的梯度不会传回 x

# .data 类似 detach（但不安全，避免使用）

# torch.no_grad() 上下文
with torch.no_grad():
    # 这里所有操作都不会构建计算图
    eval_output = model(input)
```

> **🔑 在扩散模型中**：
> - 推理时用 `torch.no_grad()` 节省显存
> - EMA 更新时用 `.data` 或 `.detach()` 避免梯度污染
> - DDIM inversion 时小心 detach 的位置

---

### 3.3 梯度裁剪

```python
# 训练循环中的标准操作
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

---

## §4 神经网络模块（nn.Module）

### 4.1 标准训练循环

```python
import torch.nn as nn
import torch.optim as optim

model = MyModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        x = batch.to(device)

        # 1. 前向
        output = model(x)
        loss = compute_loss(output)

        # 2. 反向
        optimizer.zero_grad()  # ⚠️ 必须在 backward 之前清零
        loss.backward()

        # 3. 更新
        optimizer.step()
```

**常见错误**：
- 忘记 `optimizer.zero_grad()` → 梯度累加
- 忘记 `.to(device)` → 设备不一致报错
- 在 `backward()` 后再修改张量 → 计算图错乱

---

### 4.2 自定义模块的标准写法

```python
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        # t shape: (B,)
        # 实现：见下面 sinusoidal embedding 代码
        emb = sinusoidal_embedding(t, self.dim)
        return self.mlp(emb)
```

---

### 4.3 参数与缓冲区

```python
class Schedule(nn.Module):
    def __init__(self, T):
        super().__init__()
        betas = torch.linspace(1e-4, 0.02, T)
        # ⭐ 必须用 register_buffer 而不是直接赋值
        # 否则 .to(device) 不会跟着移动！
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - betas)
        self.register_buffer('alphas_cumprod', self.alphas.cumprod(dim=0))
```

> **🔑 在扩散模型中**：beta schedule 是不可训练的常数，必须用 `register_buffer`。这是新手最常踩的坑之一。

---

## §5 数据加载

```python
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = list_files(root)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = load_image(self.files[idx])
        if self.transform:
            img = self.transform(img)
        return img

dataset = ImageDataset('./data')
loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,        # 多进程加载
    pin_memory=True,      # GPU 传输加速
    drop_last=True,       # 丢弃最后不完整 batch
)
```

---

## §6 设备与混合精度

### 6.1 设备管理

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
x = x.to(device)

# 多 GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # 简单但不推荐
# 推荐：DistributedDataParallel（DDP）或使用 accelerate
```

---

### 6.2 混合精度训练（fp16）

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in loader:
    optimizer.zero_grad()

    with autocast():               # fp16 前向
        output = model(batch)
        loss = compute_loss(output)

    scaler.scale(loss).backward()  # 缩放梯度避免下溢
    scaler.step(optimizer)         # 自动 unscale + step
    scaler.update()                # 调整 scale factor
```

**注意事项**：
- GroupNorm、Softmax 等数值敏感层可能需要保持 fp32
- bf16（PyTorch 1.10+）比 fp16 更稳定，A100/H100 上首选

---

## §7 实用代码片段（扩散模型常用）

### 7.1 Sinusoidal time embedding

```python
def sinusoidal_embedding(t, dim):
    """
    t: shape (B,), 整数时间步
    return: shape (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    embedding = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding
```

---

### 7.2 EMA 更新

```python
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(
                    v.detach(), alpha=1 - self.decay
                )

    def apply_to(self, model):
        model.load_state_dict(self.shadow)
```

---

### 7.3 按时间步索引张量

```python
def extract(a, t, x_shape):
    """
    a: shape (T,) 的 schedule 张量
    t: shape (B,) 的时间步
    x_shape: 目标张量形状（如 x.shape）
    return: shape (B, 1, 1, 1) 用于广播相乘
    """
    B = t.shape[0]
    out = a.gather(0, t)  # (B,)
    return out.view(B, *([1] * (len(x_shape) - 1)))
```

> **🔑 在扩散模型中**：每个 batch 包含不同的 $t$，需要按 $t$ 取出对应的 $\bar\alpha_t$、$\beta_t$ 等系数。这个工具函数会反复用到。

---

## §8 调试技巧

### 8.1 形状错误的快速定位

```python
# 在关键位置插入：
print(f"x.shape = {x.shape}")
assert x.shape == (B, C, H, W), f"unexpected shape: {x.shape}"
```

---

### 8.2 NaN / Inf 检测

```python
# 在 loss 计算后插入：
if not torch.isfinite(loss):
    print(f"NaN/Inf detected at step {step}")
    # 查看哪个张量出问题
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            print(f"  parameter {name} has NaN")
        if p.grad is not None and not torch.isfinite(p.grad).all():
            print(f"  gradient of {name} has NaN")
    raise RuntimeError("Training diverged")
```

---

### 8.3 显存使用查看

```python
# 训练循环中插入：
if step % 100 == 0:
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU mem: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
```

---

## §9 自查清单

完成本课程前，你应当能熟练完成下列任务而不查文档：

- [ ] 用 PyTorch 写一个完整的训练循环（model + optimizer + dataloader）
- [ ] 解释 `view` 与 `reshape` 的区别
- [ ] 解释 `requires_grad`、`detach`、`no_grad` 的区别
- [ ] 写出一个 nn.Module 子类
- [ ] 用 `register_buffer` 添加非可训练张量
- [ ] 用 GradScaler 实现 fp16 训练
- [ ] 写出 Sinusoidal embedding 的实现
- [ ] 调试 NaN 出现的位置

---

## §10 推荐资源

| 资源 | 重点 |
|------|------|
| PyTorch 官方教程 "60 minute blitz" | 入门必读 |
| PyTorch 官方 examples（GitHub） | 标准训练循环模板 |
| `lucidrains/denoising-diffusion-pytorch` | 极简扩散模型实现，代码清晰 |
| HuggingFace Diffusers 文档 | 工程级扩散模型库 |
