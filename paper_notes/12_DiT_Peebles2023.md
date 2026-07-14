# 论文导读 12: DiT (Peebles & Xie, ICCV 2023)

**标题**：Scalable Diffusion Models with Transformers
**作者**：William Peebles, Saining Xie (NYU / Berkeley)
**核心地位**：⭐⭐⭐⭐⭐ Sora / SD 3 / Flux 的架构原型

---

## 一、为什么必读

- 颠覆 UNet 在扩散模型中的统治地位
- 给出严格的 scaling law（参数 → FID）
- 直接影响 Sora、SD 3 等下一代模型设计
- ImageNet 256 SOTA：FID 2.27

---

## 二、阅读路线图

| 章节 | 优先级 |
|------|--------|
| §1 Intro | ✅ |
| §3 Diffusion Transformer | 🔴 核心 |
| §3.1 Patchify | 🔴 核心 |
| §3.2 DiT block design | 🔴 核心 |
| §4 Experimental setup | ✅ |
| §5 Experiments | 🔴 看 Fig 6 (scaling) |

---

## 三、核心架构

**Patchify** → **N × DiT blocks** → **Linear + Unpatchify**

1. Latent $z \in \mathbb{R}^{4 \times 32 \times 32}$ patchify 成 256 tokens
2. 加 2D sinusoidal positional embedding
3. 通过 N 层 Transformer blocks（每层用 AdaLN-Zero）
4. Linear projection 输出 $\epsilon$（含均值 + 协方差）
5. Unpatchify 回 latent space

---

## 四、AdaLN-Zero（论文最重要的贡献）

```python
gamma1, beta1, alpha1, gamma2, beta2, alpha2 = adaLN(c).chunk(6, dim=-1)
x = x + alpha1 * Attn(modulate(LN(x), gamma1, beta1))
x = x + alpha2 * MLP(modulate(LN(x), gamma2, beta2))
```

- 6 个调制函数（each layer）
- adaLN 权重 zero-initialized → init 时 block 是 identity

---

## 五、条件注入 4 种方式 ablation

| 方式 | FID (ImageNet 256, CFG=1.5) |
|------|------------------------------|
| In-context | 9.62 |
| Cross-attention | 7.59 |
| AdaLN | 6.81 |
| **AdaLN-Zero** | **5.55** |

AdaLN-Zero 胜出**因为 zero init 让训练稳定**，不只是 AdaLN 本身。

---

## 六、Scaling Law

模型尺寸（patch size = 2）：

| 名字 | Params | Gflops | FID (CFG=1.5) |
|------|--------|--------|---------------|
| DiT-S/2 | 33M | 1.4 | 11.20 |
| DiT-B/2 | 130M | 5.6 | 5.50 |
| DiT-L/2 | 458M | 19.7 | 3.50 |
| **DiT-XL/2** | **675M** | **29.1** | **2.27** |

**没有看到饱和**——更大的模型继续改善（这是 SD 3 8B 的依据）。

---

## 七、Patch Size Trade-off

| Patch | Tokens | FID |
|-------|--------|-----|
| 8 | 16 | 16.5 |
| 4 | 64 | 6.5 |
| **2** | **256** | **2.27** |
| 1 | 1024 | 不可行（OOM） |

$p=2$ 是性价比甜点。

---

## 八、常被误读

### 1. "DiT 取代了 SD？"
**还没**。SD 1.5/SDXL 仍是 UNet。**SD 3** 是第一个 DiT-based。DiT 的胜利是渐进的。

### 2. "DiT 必须用 AdaLN-Zero？"
**不必须**，但实证最好。SD 3 用 MMDiT（dual-stream self-attn）取代纯 AdaLN-Zero。

### 3. "DiT 比 UNet 一定快？"
**否**。同等 FID 下，DiT 在大模型规模占优；UNet 在小模型规模仍可比。计算上 attention $O(N^2)$ 在长序列贵。

---

## 九、与 ViT 的差别

DiT 不是简单照搬 ViT：
- ViT：分类，没有条件注入
- DiT：扩散，需要 t + class 条件
- DiT 的 AdaLN-Zero 是关键创新（ViT 没有）

---

## 十、思考题

1. 为什么 patch size = 2，不是 1？计算量与质量权衡
2. AdaLN-Zero 的 6 个调制函数缺一个会怎样？做 ablation
3. 把 DiT 扩展到 video：patchify 改 3D，attention 该用 spatial+temporal factorized 还是 full？
4. 推导 DiT-XL/2 在 ImageNet 256 训练 7M images 的总 FLOPs

---

## 十一、引用

```
@inproceedings{peebles2023scalable,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  booktitle={ICCV},
  year={2023}
}
```
