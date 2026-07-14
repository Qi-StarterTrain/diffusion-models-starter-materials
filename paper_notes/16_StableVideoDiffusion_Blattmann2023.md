# 论文导读 16: Stable Video Diffusion (Blattmann et al., 2023)

**标题**：Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets
**作者**：Andreas Blattmann et al. (Stability AI)
**核心地位**：⭐⭐⭐⭐ 最重要的开源 video diffusion 模型

---

## 一、为什么必读

- 开源 video diffusion 的最强 baseline（截至 2024 中）
- 详细公开了**视频数据筛选 pipeline**
- 14 帧 / 25 帧两个 checkpoint，与 SD 1.5 同生态

---

## 二、阅读路线图

| 章节 | 优先级 |
|------|--------|
| §1 Intro | ✅ |
| §2 Background | 跳过 |
| **§3 Method** | 🔴 核心 |
| §3.1 Architecture | 🔴 核心（3D UNet 改造） |
| §3.2 Stage I-III training | 🔴 核心（curriculum） |
| **§4 Data curation** | 🔴 核心（独特贡献） |
| §5 Experiments | ✅ |

---

## 三、架构

基于 SD 2.1 UNet，加入时间维：
- 每个 SD ResBlock 后插入 **temporal ResBlock**（沿时间维 1D conv）
- 每个 SD attention block 后插入 **temporal attention**
- VAE 共享 SD 的 2D VAE（不是 3D VAE）

参数量：UNet ~1.5B（比 SD 1.5 大 ~70%）。

---

## 四、训练 Curriculum（独特）

**Stage I: Image pretrain**
- 用 LAION-5B 子集训 base image model
- 等价于 SD 2.1 + 几个调整

**Stage II: Video pretrain @ 256×384**
- LVD（大规模 video 数据集，自筛选）
- 训 14 帧 video diffusion
- 14M video clips

**Stage III: Video fine-tune @ higher resolution**
- 较小但高质量的 video 数据
- 训 14 / 25 帧高清版本

→ Curriculum 让模型逐步学到时序，避免直接训 OOM。

---

## 五、数据筛选 Pipeline（论文最有价值的部分）

§4 详述：从 580M videos 筛到 152M high-quality videos。

筛选流程：
1. **Cut detection**：删掉 scene change（不能跨多个 scene 的 video）
2. **Optical flow filtering**：删掉过静态或过抖动的
3. **Text presence detection**：删掉有大量文字 overlay 的
4. **CLIP similarity to caption**：删掉文本不匹配的
5. **Aesthetics score**：删掉低美学的

最终 high-quality subset：~10M videos.

---

## 六、推理时长 & I2V 模式

SVD 默认是 **image-to-video** (I2V)：
- 输入：1 张 image
- 输出：14 帧 video
- 通过 image conditioning 注入到 UNet

可控参数：
- `motion_bucket_id` (0-255): 控制运动强度
- `fps_id`: 控制生成视频的 FPS
- `cond_aug`: 添加 noise 到 input image，增加多样性

---

## 七、实验

公开评测（FVD on UCF-101）：

| Model | FVD |
|-------|-----|
| Make-A-Video | 367 |
| Imagen Video | 296 |
| **SVD-XT (25 frames)** | **242** |
| Sora | 未公开（疑似 ~150） |

—— SVD 是当时开源 SOTA。

---

## 八、常被误读

### 1. "SVD = SD with time?"
**部分是**。SVD 加了 temporal layers，但 VAE / text encoder 与 SD 共享。Architecture 不是从头训。

### 2. "SVD 支持 text-to-video?"
**官方主推 I2V**。Text 可以隐式通过 CLIP image embedding 实现，但效果不如直接的 T2V model。

### 3. "SVD 14 帧 = 1 秒视频？"
**取决于 fps_id**。14 帧 @ 24 fps ≈ 0.6 秒；14 帧 @ 7 fps ≈ 2 秒。

---

## 九、与其他 video model 对比

| Model | 开源 | 帧数 | 分辨率 |
|-------|-----|------|--------|
| **SVD** | ✓ | 14-25 | 576×1024 |
| Open-Sora | ✓ | 16-64 | 512×512 |
| CogVideoX | ✓ | 49 | 720×480 |
| Mochi 1 | ✓ | 84 | 480p |
| Pika 1.5 | ✗ | ~100 | 1080p |
| Sora | ✗ | 600+ | 1080p |

---

## 十、思考题

1. SVD 用 2D VAE + temporal layers，而 Sora 可能用 3D VAE。两种方案的 trade-off？
2. SVD 数据筛选 pipeline 中，哪一步收益最大？设计实验验证
3. 把 SVD 的 temporal attention 改成 RoPE 3D，可能的改进？
4. 实现"SVD + ControlNet"：用 depth video 控制生成

---

## 十一、引用

```
@article{blattmann2023stable,
  title={Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets},
  author={Blattmann, Andreas and Dockhorn, Tim and Kulal, Sumith and Mendelevitch, Daniel and Kilian, Maciej and Lorenz, Dominik and Levi, Yam and English, Zion and Voleti, Vikram and Letts, Adam and others},
  year={2023}
}
```
