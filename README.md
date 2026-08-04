# Diffusion Models 科研入门培训 · 教材库

> Repo: `Qi-StarterTrain/diffusion-models-starter-materials`

> 16 周系统课程，从 DDPM 数学基础到 VLA / 具身智能。W1–W16 全部教材已写完，可跟随直播班级学习，也可直接自学。
> 本仓库是**课程教材库**（只读），后续视勘误 / 补充阅读需要更新，不再随每周课程节奏发布。
> Project 作业仓库通过 GitHub Classroom 单独下发（仅正式学员），详见下文。

---

## 课程定位

本课程面向有一定深度学习基础、希望进入扩散模型研究方向的同学。目标不是"会调用 API"，而是**能推导、能实现、能阅读前沿论文**，并最终具备独立开展相关研究的能力。

课程终点：独立实现 DDPM、DDIM、CFG、LDM 微调、Flow Matching，以及一个简化版 VLA action diffusion demo。

---

## 自学指南

本仓库已经写完 W1–W16 全部内容，可以完全自学，不必等待固定的开课节奏。建议按 P0 → P1 → P2 的顺序推进（见下方"三阶段结构"），每到一周就把该周对应的 `slides/` 讲义、`derivations/` 推导手稿、`notebooks/` 代码和 `paper_notes/` 论文导读放在一起读——讲义给直觉，推导手稿把公式一步步铺开，notebook 让你动手实现，paper note 帮你把内容接回原始论文，四者配合着看效果最好，尤其不要跳过手推公式的部分。开始之前先花一小时做 [math_prereq/self_assessment_quiz.md](math_prereq/self_assessment_quiz.md) 自测，数学基础不够扎实的话先补一补再回来，后面章节会大量依赖前面推导出的结论，跳读容易在中段卡住。每个阶段结束时用 `quizzes/` 里对应的测验（quiz1–quiz4）检验自己是不是真的掌握了，而不是"看懂了但不会推"。遇到某个点反复卡壳，先查 `supplementary/README.md`，很多课程中常见的理解难点已经被单独写成了补充笔记。至于 Project 1–5，教材库里没有配套的 starter code（那是通过 GitHub Classroom 单独下发给正式学员的），自学时可以把每个 Project 的目标当作阶段性练习，照着对应周的讲义和 notebook 自己从零搭起来。

> 本仓库服务两类读者：跟随直播班级的**正式学员**，以及只用教材库自己推进的**自学读者**。下文涉及 GitHub Classroom、助教、课程群、截止日期、考核占比的部分只对正式学员生效，标题旁会用「（正式学员）」标出；自学读者可以直接跳过这些小节，只看目录结构和分周内容即可。

---

## 提问与讨论

自学最容易卡住的地方是"看懂了但说不清哪里没懂"。遇到问题请直接在本仓库开 [Issue](https://github.com/Qi-StarterTrain/diffusion-models-starter-materials/issues)，用「提问 / 讨论」模板，标题里带上周次和涉及的讲义/推导/quiz 题号（比如 `[W5] derive_04 §4 Anderson 公式那步链式法则没看懂`），方便别人一眼判断能不能答。

- **提问前**：先搜一下已有 Issue，可能已经有人问过。
- **别只等我回复**：看到别人的 Issue，只要你能答上一部分，欢迎直接在下面回复讨论——这门课的很多"卡点"其实同学之间互相一说就通了，不需要等我来解答。
- **想到问题不一定是坏事**：Issue 不要求非得是"我错了"，也可以是"我觉得这里推导有更简洁的写法"之类的讨论。
- 我会定期看一遍所有 Issue，把还没人解决的补上，把讲得好的讨论整理进对应的 `supplementary/` 补充笔记里。

---

## 仓库结构：教材 vs. 作业（正式学员）

本课程使用**两类仓库**，请先理解它们的区别。自学读者只会用到第 1 类（本仓库），第 2 类作业库需要 GitHub Classroom 邀请，只对正式学员开放。

### 1. 教材库（本仓库）

`Qi-StarterTrain/diffusion-models-starter-materials` —— 公开只读，所有同学共用一份。

包含讲义、推导、Notebook、论文导读、数学前置等学习材料。**你不需要修改它，只需要定期 `git pull` 拿到最新内容。**

### 2. 作业库（每个 Project 一个，正式学员）

每个 Project 通过 GitHub Classroom 单独下发，接受 assignment 后会在 `Qi-StarterTrain` 组织下自动创建你的**个人作业仓库**。代码改动、TODO 实现、实验日志、最终报告都提交到该仓库。自学读者没有这一步，可参照上面"自学指南"里的建议，照着讲义和 notebook 自己实现 Project 目标。

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

| 周次 | 内容                                             | 配套材料                                  |
| ---- | ------------------------------------------------ | ----------------------------------------- |
| W9   | ControlNet · LoRA 微调                          | L10 · paper 10, 11                        |
| W10  | DiT（Diffusion Transformer）                     | L11 · paper 12 · nb08 · derive_09         |
| W11  | Flow Matching / Rectified Flow · 启动 Project 4 | L12 · paper 13, 14 · nb09 · derive_07     |
| W12  | Consistency Models                               | L13 · paper 15 · nb10 · derive_08         |
| W13  | Video Diffusion                                  | L14 · paper 16 · nb11 · derive_10         |
| W14  | Sora 解剖 · Quiz 3（覆盖 W10–W14）             | L15 · paper 17                            |
| W15  | World Models                                     | L16                                       |
| W16  | VLA / 具身智能 · 启动 Project 5 · Quiz 4（覆盖 W15–W16） | L17 · paper 18, 19 · nb12        |

---

## 本仓库目录

```
diffusion-models-starter-materials/
├── math_prereq/          ← 自测 + 数学速查
│   ├── self_assessment_quiz.md     （15 题自测，开始前必做）
│   ├── self_assessment_answers.md  （自测答案）
│   ├── prob_review.md              （概率论速查）
│   ├── calculus_review.md          （微积分 / 线代速查）
│   └── pytorch_primer.md           （PyTorch 速查）
│
├── slides/               ← 各周讲义（L01–L17，已全部写完）
├── derivations/          ← 公式推导手稿（derive_01–10，已全部写完）
├── notebooks/            ← 教学 Notebook（nb01–nb12，已全部写完）
├── paper_notes/          ← 论文导读（01–19，已全部写完）
├── quizzes/              ← 阶段测验与答案（quiz1–quiz4，每个含 questions.md / answer_key.md）
├── supplementary/        ← 补充阅读材料，总入口见 supplementary/README.md
└── templates/            ← 实验日志 / 论文精读笔记模板，做 Project 或读论文时可直接套用
```

本仓库已经包含 P0、P1、P2 全部三个阶段的教材内容（W1–W16），讲义、推导、Notebook、paper notes 与 quizzes 都按主题合并在根目录对应文件夹中，不再分阶段单独存放。

### 当前已入库阶段

- `math_prereq/`、`templates/`、`supplementary/` 是跨阶段共用材料。
- `slides/`（L01–L17）、`derivations/`（derive_01–10）、`notebooks/`（nb01–nb12）、`paper_notes/`（01–19）、`quizzes/`（quiz1–quiz4）已覆盖 W1-W16 全部三个阶段的主线内容。
- Project 对应的 starter code（含 Project 4 Flow Matching、Project 5 VLA Action Diffusion）仍以各自的 Classroom / template 仓库为准（正式学员），不在本教材库长期维护；自学读者参照"自学指南"自行实现即可。

补充阅读材料的总入口见 [supplementary/README.md](supplementary/README.md)，其中 L06 Score SDE 相关的数学基础建议按“ODE/SDE 数值方法 → Continuous Schedule → Probability Flow ODE”顺序阅读。

---

## 开始之前

### 第一步：克隆教材库

```bash
git clone https://github.com/Qi-StarterTrain/diffusion-models-starter-materials.git
cd diffusion-models-starter-materials
```

教材已全部写完，`git pull` 主要用于同步后续的勘误和补充阅读更新。

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

### 第四步：接受第一个 Project assignment（正式学员）

W3 课程结束时，助教会在课程群里发 Classroom 邀请链接。点击后会在 `Qi-StarterTrain` 组织下为你自动创建 Project 1 的个人作业仓库。后续 Project 同理。自学读者没有这一步，直接参照上面"自学指南"里的做法，照着讲义和 notebook 自己实现每个 Project 的目标即可。

---

## 推荐目录布置（正式学员）

建议在本地用如下结构组织，避免把两类仓库混在一起（自学读者只有 `materials/`，不需要 `assignments/`）：

```
~/diffusion-course/
├── materials/                       ← git clone 本仓库到这里
└── assignments/
    ├── project1-ddpm-<你的用户名>/   ← Classroom 自动创建
    ├── project2-samplers-<你的用户名>/
    └── ...
```

---

## 考核方式（正式学员）

自学读者没有正式考核，可以把 quiz 当作自我检验：每个阶段学完后限时做一遍，答不上来的地方回去重读对应讲义 / 推导。

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

- 教材内容 W1–W16 已全部写完，后续只做勘误和补充阅读更新，请养成定期 `git pull` 的习惯。
- 作业仓库由 Classroom 独立创建（仅正式学员），**不会**自动同步教材更新；如遇 README 等不一致，以本教材库为准。
- 发现公式错误或代码 bug 请直接提 Issue 或联系我，优先级最高。
- 课程设计强调**推导 > 记忆**，所有公式都有完整推导过程，不存在"魔法系数"。
