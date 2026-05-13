# Diffusion Models 科研入门培训

> 16 周系统课程，从 DDPM 数学基础到 VLA / 具身智能。
> 本仓库为学生开放版，随课程进度逐步更新。

---

## 课程定位

本课程面向有一定深度学习基础、希望进入扩散模型研究方向的同学。目标不是"会调用 API"，而是**能推导、能实现、能阅读前沿论文**，并最终具备独立开展相关研究的能力。

课程终点：独立实现 DDPM、DDIM、CFG、LDM 微调、Flow Matching，以及一个简化版 VLA action diffusion demo。

---

## 三阶段结构

| 阶段         | 周次    | 主题                                                                       | 项目                                                            |
| ------------ | ------- | -------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **P0** | W1–W3  | 数学基础 · DDPM                                                           | Project 1：从零实现 DDPM                                        |
| **P1** | W4–W8  | Score SDE · DDIM · CFG · LDM/SD                                         | Project 2：采样器对比`<br>`Project 3：Stable Diffusion 解剖   |
| **P2** | W9–W16 | DiT · Flow Matching · Consistency · Video · Sora · World Model · VLA | Project 4：Flow Matching`<br>`Project 5：VLA Action Diffusion |

---

## 分周安排

### P0：数学基础 + DDPM（W1–W3）

| 周次  | 内容                                               |
| ----- | -------------------------------------------------- |
| W1    | 生成模型导论 · VAE · Score Matching 入门         |
| W2    | DDPM Forward Process（重点：手推公式）             |
| W3    | DDPM Training / U-Net / Sampling · 启动 Project 1 |
| W4 末 | **Project 1 截止** + Quiz 1                  |

### P1：核心方法（W4–W8）

| 周次 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| W4   | Improved DDPM（Cosine schedule · learned variance）         |
| W5   | Score SDE（VP/VE-SDE · probability flow ODE，课程数学高峰） |
| W6   | DDIM + DPM-Solver · 启动 Project 2                          |
| W7   | Classifier Guidance · CFG · Cross-Attention                |
| W8   | LDM / Stable Diffusion 工程解剖 · 启动 Project 3 · Quiz 2  |

### P2：前沿进阶（W9–W16）

| 周次 | 内容                                             |
| ---- | ------------------------------------------------ |
| W9   | ControlNet · LoRA 微调                          |
| W10  | DiT（Diffusion Transformer）                     |
| W11  | Flow Matching / Rectified Flow · 启动 Project 4 |
| W12  | Consistency Models                               |
| W13  | Video Diffusion                                  |
| W14  | Sora 解剖 · Quiz 3                              |
| W15  | World Models                                     |
| W16  | VLA / 具身智能 · 启动 Project 5 · Quiz 4       |

---

## 本仓库内容

```
diffusion-models-starter/
├── math_prereq/          ← 开课前自测 + 数学速查
│   ├── self_assessment_quiz.md   （15 题，开课前必做）
│   ├── prob_review.md            （概率论速查）
│   ├── calculus_review.md        （微积分 / 线代速查）
│   └── pytorch_primer.md         （PyTorch 速查）
│
├── slides/               ← 各周讲义（随课程发布）
├── derivations/          ← 公式推导手稿（随课程发布）
├── notebooks/            ← 教学 Notebook（随课程发布）
└── paper_notes/          ← 论文导读（随课程发布）
```

Project 脚手架代码、Quiz 题库由助教单独下发，不在本仓库中。

---

## 开始之前

**第一步：完成数学自测**

打开 [math_prereq/self_assessment_quiz.md](math_prereq/self_assessment_quiz.md)，独立完成 15 题（约 1 小时）。

- 全部正确 → 直接进入课程
- 答错 4–7 题 → 阅读对应 review 材料补强
- 答错 ≥8 题 → 建议先系统补数学再回来

**第二步：配置环境**

```bash
pip install torch torchvision diffusers transformers peft accelerate \
            torchmetrics scikit-learn matplotlib tqdm PyYAML jupyter
```

建议 Python 3.10+，torch ≥ 2.0。

---

## 考核方式

| 项目                       | 时间     | 占比（参考） |
| -------------------------- | -------- | ------------ |
| Project 1（DDPM）          | W3–W4   | 20%          |
| Project 2/3（采样器 / SD） | W6–W8   | 30%          |
| Project 4/5（FM / VLA）    | W11–W16 | 30%          |
| Quiz 1–4                  | 各阶段末 | 20%          |

---

## 学习资源

课程全部材料以 Markdown 格式维护，方便版本管理。核心论文按课程顺序整理为导读笔记，格式统一：阅读路线图 · 核心公式 · 常被误读 · 思考题。

涉及的主要论文（按课程顺序）：

DDPM · Improved DDPM · Score SDE · DDIM · Diffusion Beats GAN · CFG · LDM/SD · DPM-Solver · EDM · ControlNet · LoRA · DiT · Flow Matching · Rectified Flow · Consistency Models · SVD · Sora · OpenVLA · Pi-0

---

## 说明

- 材料随每周课程内容在本仓库更新，请保持 `git pull`。
- 发现公式错误或代码 bug 请直接联系我，优先级最高。
- 课程设计强调**推导 > 记忆**，所有公式都有完整推导过程，不存在"魔法系数"。
