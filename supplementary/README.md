# 补充阅读材料

本目录整理补充阅读材料，内容用于配合扩散模型入门课程的数学直觉、公式推导和生成模型背景阅读。

## Score SDE 数学基础建议顺序

如果是为了理解 L06 Score SDE，可以按这个顺序读：

1. [ODE 与 SDE 数值方法入门](ode-sde-numerical-methods.md)：先补 Taylor、Euler、误差阶、Euler-Maruyama。
2. [Continuous Schedule 与 SDE 视角](continuous-schedule-sde-view.md)：理解 $\beta_t=\beta(s)\Delta t$ 和 DDPM 到 VP-SDE 的连续极限。
3. [Probability Flow ODE：如何理解“边缘分布完全相同”](probability-flow-ode-marginals.md)：理解 SDE 与 ODE 为什么能共享同一组边缘分布，以及 DDIM / DPM-Solver 的 ODE 视角。

## 阅读材料

- [VAE 补充阅读材料](vae-supplementary-reading.md)
- [Flow-based Model 理解](flow-based-model.md)
- [Score Function 几何直觉解读](score-function-geometric-intuition.md)
- [Score Matching 解读](score-matching-explained.md)
- [高斯 Score 推导：为什么预测噪声等价于预测 Score](gaussian-score-noise-prediction.md)
- [朗之万动力学理解](langevin-dynamics.md)
- [SNR 信噪比的重要性理解](snr-importance.md)
- [DDPM 方差稳定性推导](ddpm-variance-stability.md)
- [Continuous Schedule 与 SDE 视角](continuous-schedule-sde-view.md)
- [Probability Flow ODE：如何理解“边缘分布完全相同”](probability-flow-ode-marginals.md)
- [ODE 与 SDE 数值方法入门](ode-sde-numerical-methods.md)
- [EMA 参数平均的作用](ema-for-diffusion.md)
- [独立随机变量相加方差推导](variance-of-independent-sum.md)
- [维度灾难理解](curse-of-dimensionality.md)
