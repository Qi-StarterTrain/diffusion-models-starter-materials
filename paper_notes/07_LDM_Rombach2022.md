# 论文导读 07: LDM / Stable Diffusion (Rombach et al., CVPR 2022)

**标题**：High-Resolution Image Synthesis with Latent Diffusion Models
**作者**：Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (CompVis)
**核心地位**：⭐⭐⭐⭐⭐ **改变扩散模型工业格局的论文**。Stable Diffusion 的源头

---

## 一、为什么必读

- 把扩散从像素空间搬到 latent 空间，**48x 减少计算**
- 让"消费级显卡训练扩散"成为可能
- Stable Diffusion 1.x/2.x/3.x 都基于此
- 在 4 大任务上 SOTA：unconditional, super-resolution, inpainting, text-to-image

> 不读 LDM 就不懂为什么 SD 是 SD。

---

## 二、阅读路线图

| 章节 | 优先级 |
|------|--------|
| §1 Intro | ✅ |
| §2 Related Work | 跳过 |
| **§3 Method** | 🔴 核心 |
| §3.1 Perceptual image compression | 🔴 核心（VAE 设计） |
| §3.2 LDM | 🔴 核心 |
| §3.3 Conditioning mechanisms | 🔴 核心（cross-attention） |
| §4 Experiments | ✅ |
| §4.1-§4.4 | ✅ |

---

## 三、核心架构

```
x → [Encoder] → z → [LDM (UNet)] → z' → [Decoder] → x'
     |                |                |
     固定预训练       条件 y 通过        固定预训练
                    cross-attn 注入
```

---

## 四、§3.1 VAE 设计（必看）

### 损失
$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{LPIPS}} + \lambda_{\text{adv}} \mathcal{L}_{\text{GAN}} + \lambda_{\text{KL}} D_{\text{KL}}$$

- **Recon**: L1 pixel loss
- **LPIPS** (Zhang 2018): VGG perceptual loss
- **GAN**: PatchGAN discriminator，让高频细节锐利
- **KL**: 极小权重 ($\lambda_{\text{KL}} \approx 10^{-6}$)

### 设计选择
- **VQ-VAE vs KL-VAE**：实验显示 KL-VAE 更适合 LDM（连续 latent + diffusion 兼容）
- **Downsample factor**：$f = 4, 8, 16$。SD 1.5 用 $f=8$（4 channels latent）
- **Latent channels**：4（SD 1.5/2.0），16（SD 3）

---

## 五、§3.2 LDM（必看）

训练：
1. $x \xrightarrow{E} z_0$（一次性）
2. $z_0 \xrightarrow{加噪} z_t$
3. UNet 预测噪声：$\epsilon_\theta(z_t, t, c)$
4. Loss = MSE

推理：
1. $z_T \sim \mathcal{N}(0, I)$
2. 50 步 DDIM 采样得到 $z_0$
3. $z_0 \xrightarrow{D} x$

---

## 六、§3.3 Cross-Attention（必看）

```
Q from image features (B, HW, C_img)
K, V from condition embeddings (B, T_cond, C_cond)
Output: weighted V (each position 'looks at' condition)
```

注入位置：每个 ResBlock 后 → 在多个分辨率（32, 16, 8）都注入。

---

## 七、关键工程细节

### 1. Scaling Factor 0.18215
SD 1.5 的 latent **不是真正的高斯**：训练后实测 latent std ≈ 5.5。乘 0.18215 让 scaled latent std ≈ 1，匹配 DDPM 假设。

### 2. CLIP Text Encoder
SD 1.x 用 CLIP ViT-L/14（冻结）。SD 2.x 改用 OpenCLIP ViT-H/14（社区接受度低）。SD 3 用 T5 + CLIP 三个 encoder 混合。

### 3. UNet 参数量
SD 1.x: ~860M。比像素空间 DDPM 的 ~50M 大得多——因为 LDM 把"计算预算"从加噪步数转移到了模型容量。

---

## 八、容易误读

### 1. "Latent space 是 VAE 的 z?"

是，**但和经典 VAE 不同**：
- KL 权重极小，latent 不严格是 $\mathcal{N}(0, I)$
- LDM 在 latent 上做 DDPM 假设的"$z_T \sim \mathcal{N}(0, I)$" 严格上不成立
- 实践中由于 forward 加噪足够，影响不大

---

### 2. "LDM 适用于所有任务？"

**有限制**：
- VAE 重构会**丢失细节**（特别是文字、小目标）
- 像素级精确任务（segmentation）不适合 latent diffusion
- 高动态范围（HDR）受 VAE 限制

SD 3 用更大的 VAE（16 channels）部分缓解。

---

### 3. "LDM = SD？"

**LDM 是方法，SD 是模型**。
- LDM 是 Rombach 等人在 LAION-400M 上训练的产物
- "Stable Diffusion" 是 Stability AI / CompVis 合作发布的具体 checkpoint 系列

---

## 九、关键实验表

§4.1 Table 8: $f$ 选择的 FID
| f | FID (CelebA-HQ) | 计算量 |
|---|-----------------|--------|
| 1 (像素) | 27.4 | 1x |
| 4 | 14.0 | 16x 减少 |
| **8** | **15.5** | **64x 减少** |
| 16 | 28.7 | 256x 但质量降 |

—— $f=8$ 是甜点（质量与计算的最佳折中）。

---

## 十、与其他论文的关系

- **DDPM, Improved DDPM, Score SDE**：LDM 在 latent 上跑 DDPM
- **VQGAN (Esser 2021)**：同一作者的前作，提供 VAE 设计
- **DiT (Peebles 2023)**：用 Transformer 替换 UNet 的 LDM
- **SD 1/2/3/Flux**：LDM 的工业化产物

---

## 十一、思考题

1. 如果 KL 权重 $\lambda$ 增大到 1（标准 VAE），SD 会出什么问题？
2. 为什么 SD 不能生成清晰文字？从 VAE 重构 loss 分析
3. SDXL 的 latent 大小是 (4, 128, 128)（1024×1024 输出）。这与 SD 1.5 的 (4, 64, 64) 对比，UNet 计算量约多少倍？

---

## 十二、引用

```
@inproceedings{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={CVPR},
  year={2022}
}
```
