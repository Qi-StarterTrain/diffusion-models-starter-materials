# Diffusion Models 科研入门培训 · 教材库

> Repo: `Qi-StarterTrain/diffusion-models-starter-materials`

> 16 周系统课程，从 DDPM 数学基础到 VLA / 具身智能。
> 本仓库是**课程教材库**（只读），随课程进度持续更新。
> Project 作业仓库通过 GitHub Classroom 单独下发，详见下文。

---

## 课程定位

本课程面向有一定深度学习基础、希望进入扩散模型研究方向的同学。目标不是"会调用 API"，而是**能推导、能实现、能阅读前沿论文**，并最终具备独立开展相关研究的能力。

课程终点：独立实现 DDPM、DDIM、CFG、LDM 微调、Flow Matching，以及一个简化版 VLA action diffusion demo。

---

## 仓库结构：教材 vs. 作业

本课程使用**两类仓库**，请先理解它们的区别：

### 1. 教材库（本仓库）

`Qi-StarterTrain/diffusion-models-starter-materials` —— 公开只读，所有同学共用一份。

包含讲义、推导、Notebook、论文导读、数学前置等学习材料。**你不需要修改它，只需要定期 `git pull` 拿到最新内容。**

### 2. 作业库（每个 Project 一个）

每个 Project 通过 GitHub Classroom 单独下发，接受 assignment 后会在 `Qi-StarterTrain` 组织下自动创建你的**个人作业仓库**。代码改动、TODO 实现、实验日志、最终报告都提交到该仓库。

| Project | Classroom assignment 状态 |
| ------- | ------------------------- |
| Project 1：DDPM from scratch | W3 开放 |
| Project 2：采样器对比 | W6 开放 |
| Project 3：Stable Diffusion 解剖 | W8 开放 |
| Project 4：Flow Matching | W11 开放 |
| Project 5：VLA Action Diffusion | W14 开放 |

> **重要**：作业库是从 template fork 出来的，**不会自动同步本教材库的更新**。教材类内容请始终以本仓库 `main` 分支为准。

---

## 三阶段结构

| 阶段 | 周次 | 主题 | 项目 |
| ---- | ---- | ---- | ---- |
| **P0** | W1–W3 | 数学基础 · DDPM | Project 1：从零实现 DDPM |
| **P1** | W4–W8 | Score SDE · DDIM · CFG · LDM/SD | Project 2：采样器对比<br>Project 3：Stable Diffusion 解剖 |
| **P2** | W9–W16 | DiT · Flow Matching · Consistency · Video · Sora · World Model · VLA | Project 4：Flow Matching<br>Project 5：VLA Action Diffusion |

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

## 本仓库目录

```
diffusion-models-starter-materials/
├── math_prereq/          ← 开课前自测 + 数学速查
│   ├── self_assessment_quiz.md   （15 题，开课前必做）
│   ├── prob_review.md            （概率论速查）
│   ├── calculus_review.md        （微积分 / 线代速查）
│   └── pytorch_primer.md         （PyTorch 速查）
│
├── slides/               ← 各周讲义（随课程发布）
├── derivations/          ← 公式推导手稿（随课程发布）
├── notebooks/            ← 教学 Notebook（随课程发布）
├── paper_notes/          ← 论文导读（随课程发布）
└── supplementary/        ← 补充阅读材料
```

Project 脚手架代码、Quiz 题库不在本仓库——分别通过 Classroom assignment 和助教单独下发。

---

## 开始之前

### 第一步：克隆教材库

```bash
git clone https://github.com/Qi-StarterTrain/diffusion-models-starter-materials.git
cd diffusion-models-starter-materials
```

之后每周开课前跑一次 `git pull` 即可拿到新讲义、新推导、新 notebook。

### 第二步：完成数学自测

打开 [math_prereq/self_assessment_quiz.md](math_prereq/self_assessment_quiz.md)，独立完成 15 题（约 1 小时）。

- 全部正确 → 直接进入课程
- 答错 4–7 题 → 阅读对应 review 材料补强
- 答错 ≥8 题 → 建议先系统补数学再回来

### 第三步：配置环境

```bash
pip install torch torchvision diffusers transformers peft accelerate \
            torchmetrics scikit-learn matplotlib tqdm PyYAML jupyter
```

建议 Python 3.10+，torch ≥ 2.0。

### 第四步：接受第一个 Project assignment

W3 课程结束时，助教会在课程群里发 Classroom 邀请链接。点击后会在 `Qi-StarterTrain` 组织下为你自动创建 Project 1 的个人作业仓库。后续 Project 同理。

---

## 推荐目录布置

建议在本地用如下结构组织，避免把两类仓库混在一起：

```
~/diffusion-course/
├── materials/                       ← git clone 本仓库到这里
└── assignments/
    ├── project1-ddpm-<你的用户名>/   ← Classroom 自动创建
    ├── project2-samplers-<你的用户名>/
    └── ...
```

---

## 考核方式

| 项目                       | 时间     | 占比（参考） |
| -------------------------- | -------- | ------------ |
| Project 1（DDPM）          | W3–W4   | 20%          |
| Project 2/3（采样器 / SD） | W6–W8   | 30%          |
| Project 4/5（FM / VLA）    | W11–W16 | 30%          |
| Quiz 1–4                  | 各阶段末 | 20%          |

---

## 涉及论文（按课程顺序）

DDPM · Improved DDPM · Score SDE · DDIM · Diffusion Beats GAN · CFG · LDM/SD · DPM-Solver · EDM · ControlNet · LoRA · DiT · Flow Matching · Rectified Flow · Consistency Models · SVD · Sora · OpenVLA · Pi-0

每篇都会在课程中以 paper note 形式给出导读：阅读路线图 · 核心公式 · 常被误读 · 思考题。

---

## 说明

- 教材内容**每周更新**，请养成开课前 `git pull` 的习惯。
- 作业仓库由 Classroom 独立创建，**不会**自动同步教材更新；如遇 README 等不一致，以本教材库为准。
- 发现公式错误或代码 bug 请直接提 Issue 或联系我，优先级最高。
- 课程设计强调**推导 > 记忆**，所有公式都有完整推导过程，不存在"魔法系数"。
