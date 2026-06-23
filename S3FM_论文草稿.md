# Sparse-Sensor-Guided Flow Matching for Efficient Spatiotemporal Dynamics Reconstruction

> 论文草稿 v0。本文档用于建立完整认知和记录当前研究状态，不等同于最终投稿稿件。
>
> 当前定位：**问题定义 + 方法主线 + 创新边界 + 实验设计 + 已有证据 + 后续工作**。

## 0. 摘要草稿

本文研究如何从稀疏、不完整或带噪的传感器观测中重建完整时空动力学场。已有 S3GM 框架通过预训练生成先验和推理时测量引导实现了算子无关的稀疏重建，但其核心采样过程依赖约 1000 步 reverse-SDE，在高维时空场上计算代价较高。本文提出 S3FM：一种面向稀疏传感器逆问题的 flow-matching 重建框架。S3FM 保留 S3GM 的“预训练先验 + 推理时观测引导”结构，但将采样过程改为少步 ODE，并进一步引入信息化源分布，使流从一个由观测推断出的低带宽近似场出发，而非从纯高斯噪声出发。

核心假设是：真正的加速不应仅归因于 SDE 到 ODE 的替换，而应来自信息化源带来的更短传输路径。为验证这一点，本文设计了 Gaussian-source S3FM、learned-source S3FM、以及 S3GM probability-flow ODE 等归因基线。当前 KSE 实验表明，在 15% sparse observation 下，learned-source S3FM 在 10-100 NFE 范围内稳定优于 Gaussian-source S3FM 和 validation-tuned S3GM-PF-ODE 基线，并且 learned source 到目标场的位移显著短于 marginal source。这支持了“信息化源缩短传输路径，从而实现少步重建”的方法解释。

## 1. 核心问题

目标是从稀疏测量恢复完整时空动力学场：

```math
y = H(X) + \epsilon
```

其中：

- `X` 是完整时空场；
- `H` 是已知且可微的测量算子；
- `y` 是稀疏、不完整或带噪观测；
- `\epsilon` 是测量噪声。

我们希望恢复完整 `X`，并在更长序列中支持重建和预测。

本文关注的问题不是简单地“用 flow matching 替代 diffusion”，而是：

> 一个从信息化源分布出发的预训练 flow-matching 先验，能否在不针对每个测量算子 `H` 重新训练的前提下，以远少于 S3GM 的函数评估次数完成稀疏时空重建？并且这种加速能否被归因于更短的传输路径，而不是仅仅来自 ODE 采样器？

这个问题包含三个关键约束：

1. **算子无关性**：预训练先验不依赖具体 `H`，不同 sparse mask 或测量形式只在推理时通过 guidance 进入。
2. **少步推理**：目标是在 10-50 NFE 内获得有效重建，而不是依赖 1000 步采样。
3. **归因清晰**：必须证明优势不只是 “SDE 换 ODE”，而是信息化源和短传输路径带来的。

## 2. 背景与动机

S3GM 的核心价值在于分离了两类信息：

1. 一个预训练时空生成先验，负责建模“什么是合理的动力学场”。
2. 推理时测量引导，负责把当前观测 `y,H` 加入采样过程。

这种分离让 S3GM 能够适配不同传感器布局和测量形式，但代价是 reverse-SDE 采样通常需要大量步数。对于高维时空场，1000 步采样会成为推理瓶颈。

S3FM 的基本想法是保留 S3GM 的算子无关后验重建范式，同时将采样过程改写为 flow-matching ODE：

```text
S3GM: pretrained score prior + measurement-guided reverse SDE
S3FM: pretrained velocity prior + measurement-guided forward ODE
```

但这还不够。标准 Gaussian-source flow matching 在数学上与 diffusion probability-flow ODE 高度接近，本身不能构成强方法贡献。因此，本文把真正的创新放在 **informative source / learned source** 上：让 flow 从一个与目标相关、低带宽、观测可推断的 source 出发，而不是从纯高斯噪声出发。

## 3. 方法概述

S3FM 使用条件 flow matching 训练一个速度场先验：

```math
X_s = (1-s)X_0 + sX_1
```

```math
\mathcal{L}_{FM}
= \mathbb{E}\left[
\|v_\theta(X_s,s) - (X_1-X_0)\|^2
\right]
```

其中：

- `X_1` 是 clean target；
- `X_0` 是 source sample；
- `s \in [0,1]` 是 flow time；
- `v_\theta` 是预训练 velocity model。

推理时通过 endpoint estimate 得到 clean field 估计：

```math
\hat{X}_1 = X_s + (1-s)v_\theta(X_s,s)
```

然后用观测一致性能量引导 ODE：

```math
J_{obs} = \|y - H(\hat{X}_1)\|^2
```

```math
\frac{dX_s}{ds}
= v_\theta(X_s,s) - \lambda(s)\nabla_{X_s}J_{obs}(\hat{X}_1)
```

这使得模型在保持预训练先验不变的情况下，通过推理时 guidance 适配不同观测算子。

## 4. 关键创新

### 4.1 信息化源分布

Gaussian-source flow matching 从纯噪声出发：

```math
X_0 \sim \mathcal{N}(0,I)
```

这与 diffusion probability-flow ODE 在本质上非常接近，因此不能作为主要贡献。本文的核心贡献是 learned informative source：

```math
X_0 = q_\phi(y,H)
```

其中 `q_phi` 从稀疏观测和测量算子推断一个低带宽、观测相关的 source。它不是完整重建器，而是为 flow 提供一个更接近目标的起点。

直观上，Gaussian source 要从纯噪声“生成整张场”，而 learned source 已经包含目标的大尺度结构，flow 只需补充缺失细节。因此传输距离更短，少步 ODE 更容易成功。

### 4.2 算子无关的 flow guidance

S3FM 不把 `H` 编进先验训练，而是在推理时通过可微 energy guidance 加入观测约束。这继承了 S3GM 的灵活性：只要 `H` 可微，就可以在不重新训练先验的情况下更换测量形式。

当前主方法使用 covariance-gradient guidance，即让梯度穿过：

```text
X_s -> v_theta(X_s,s) -> X1_hat -> J_obs -> grad with respect to X_s
```

因此它不是简单对 endpoint 做后处理，而是在 ODE 积分过程中持续改变速度场，使轨迹朝观测一致方向移动。

### 4.3 归因基线：S3GM-PF-ODE

为了避免把优势错误归因于 flow matching，本文加入 S3GM probability-flow ODE 基线。该基线同样使用确定性 ODE，并对 guidance lambda 做 validation tuning。

只有当 learned-source S3FM 在相同 NFE 下优于 S3GM-PF-ODE，才能说明优势不只是 ODE 采样器带来的。

### 4.4 短传输路径机制

本文的机制解释是：

```text
Gaussian source:
  X0 是纯噪声，离目标 X1 很远，flow 需要生成完整结构。

Learned source:
  X0 已经包含目标的大尺度结构，离 X1 更近，flow 主要补充细节。
```

因此，在相同 NFE 下，learned-source flow 更容易被少步 ODE 精确积分，也更容易被观测 guidance 推向正确解。

## 5. 实验设计

当前主要验证系统是 Kuramoto-Sivashinsky equation (KSE)。实验使用 sparse observation，从完整时空场中随机观测一小部分点，目标是恢复完整窗口。

主要比较对象：

1. **S3FM-Gauss**：高斯源 flow matching，对照“普通 FM 是否足够”。
2. **S3FM learned-source**：从 `q_phi(y,H)` 推断的信息化源出发。
3. **S3GM-PF-ODE**：score prior 的 probability-flow ODE 版本，用于隔离 ODE-vs-SDE 因素。
4. **Unguided variants**：验证 measurement guidance 是否确实贡献重建质量。

主要指标：

- full-field nRMSE；
- observation residual；
- learned source 到目标的 displacement；
- 不同 NFE 下的速度-精度曲线；
- 不同 mask seed / source seed 下的稳定性。

## 6. 当前实验结果记录

### 6.1 S3FM-Gauss guidance 可工作

在 15% sparse observation 下，单窗口 S3FM-Gauss 的 `g_cov-G` guidance 能显著降低 nRMSE 和 observation residual。已有 sweep 显示：

| lambda | nRMSE | obs_resid |
|---:|---:|---:|
| 0 | 1.284 | 1.284 |
| 1.0 | 0.502 | 0.462 |
| 2.0 | 0.237 | 0.207 |
| 5.0 | 0.083 | 0.070 |
| 10.0 | 0.044 | 0.035 |

这说明 flow guidance 的符号、梯度路径和观测一致性机制已经跑通。

### 6.2 Learned-source 明显优于 Gaussian-source

在 64 test windows、source seeds 0/1/2、mask seeds 0/1/2、observed fraction 0.15 下，learned-source S3FM 在所有 mask seed 和 NFE 上通过 M4b gate。gate 要求：

- learned Info guided <= Gaussian guided；
- learned Info guided <= learned Info unguided。

代表性结果：

| mask seed | NFE | Gaussian guided | learned Info guided | learned Info unguided |
|---:|---:|---:|---:|---:|
| 0 | 10 | 0.2521 | 0.0900 | 0.0942 |
| 1 | 10 | 0.2816 | 0.0967 | 0.1003 |
| 2 | 10 | 0.2364 | 0.0928 | 0.0946 |

这说明 learned source 不是简单增加复杂度，而是在低 NFE 下提供了稳定收益。

### 6.3 Learned source 的传输路径更短

机制诊断显示 learned source 到目标场的 displacement 远小于 marginal source：

| mask seed | marginal displacement | warm displacement | learned displacement | learned source nRMSE |
|---:|---:|---:|---:|---:|
| 0 | 47.8945 | 2.6269 | 3.2565 | 0.1121 |
| 1 | 47.8945 | 2.7434 | 3.4588 | 0.1175 |
| 2 | 47.8945 | 2.7309 | 3.1735 | 0.1119 |

这支持本文的核心机制解释：信息化源把起点移到目标附近，使 flow 的传输任务变短。

### 6.4 S3GM-PF-ODE 归因基线

在 validation-tuned PF lambda 设置下，S3GM-PF-ODE 仍显著弱于 learned-source S3FM：

| NFE | PF lambda | S3GM-PF guided nRMSE | S3FM learned guided nRMSE |
|---:|---:|---:|---:|
| 10 | 0.1 | 1.0384 +- 0.0113 | 0.0900 |
| 20 | 0.25 | 0.8657 +- 0.0088 | 0.0907 |
| 50 | 0.5 | 0.6432 +- 0.0073 | 0.0914 |
| 100 | 1.0 | 0.4173 +- 0.0037 | 0.0916 |

这初步排除了“只是因为 ODE 比 SDE 快”的解释。不过当前 PF-ODE 是内部 score-prior attribution baseline，不是原 S3GM 作者发布模型的严格复现，最终论文中需要明确标注。

### 6.5 Server suite smoke

当前已有一个极小端到端 smoke suite：

- observed fraction: 0.15；
- mask seed: 0；
- source seed: 0；
- num windows: 8；
- NFE: 10；
- PF lambda sweep: 0, 0.1, 0.25。

该 smoke suite 已完成：

- M4b learned-source；
- PF validation lambda sweep；
- PF test with validation-selected lambda；
- aggregate CSV / README 汇总。

结果显示：

- M4b learned-source pass: 1/1；
- val-tuned PF attribution pass: 1/1。

这说明服务器 suite 的端到端程序链路可用，但它不替代正式实验。

## 7. 论文主张

本文可以主张：

1. S3FM 将 S3GM 的 sparse-sensor posterior reconstruction 框架迁移到 flow-matching ODE 中。
2. 标准 Gaussian-source FM 不是主要贡献，因为它和 diffusion/PF-ODE 有强等价关系。
3. 真正贡献是 learned informative source，使 flow 从观测相关、低带宽、接近目标的 source 出发。
4. learned-source S3FM 在 KSE sparse reconstruction 中以很低 NFE 获得显著优于 Gaussian-source 和 PF-ODE attribution baseline 的重建质量。
5. displacement 诊断支持短传输路径解释。

## 8. 需要谨慎表述的边界

不能过度声称：

- 不能说“flow matching 本身天然优于 diffusion”。
- 不能把 Gaussian-source S3FM 当作核心创新。
- 不能声称已经完整复现原始 S3GM，除非之后补上严格复现实验。
- 不能把 learned source 写成 oracle；它必须来自 `y,H`，而不是 `X_true`。
- 不能说已经完成所有论文级实验；当前服务器 suite 只是端到端可用，正式大规模实验仍需跑完。

## 9. 后续实验计划

正式论文还需要补齐：

1. 服务器 full suite：更多 windows、mask seeds、observed fractions。
2. observed fraction sweep：例如 0.05 / 0.10 / 0.15 / 0.30。
3. NFE sweep：10 / 20 / 50 / 100。
4. 更完整的 PF-ODE attribution baseline。
5. wall-clock time 对比，证明低 NFE 不只是指标好，也确实更快。
6. 若时间允许，加入更接近原始 S3GM 的复现或公开 checkpoint 对比。

## 10. 建议论文结构

### 10.1 Introduction

- 稀疏传感器时空重建的重要性；
- S3GM 的优势：预训练生成先验 + 推理时 measurement guidance；
- S3GM 的瓶颈：1000-step reverse-SDE；
- 本文问题：如何在保持算子无关性的同时实现少步重建；
- 本文观点：关键不是 FM 替代 diffusion，而是 informative source 缩短传输。

### 10.2 Related Work

- Spatiotemporal generative priors；
- Sparse-sensor reconstruction；
- Score-based inverse problems / S3GM；
- Flow matching；
- Guidance for flow matching。

### 10.3 Method

- Problem formulation；
- Base flow-matching prior；
- Endpoint estimate；
- Measurement guidance；
- Learned informative source；
- Training and inference pipeline；
- Difference from Gaussian-source FM and diffusion/PF-ODE。

### 10.4 Experiments

- KSE setup；
- Sparse observation setup；
- Baselines；
- Metrics；
- S3FM-Gauss guidance validation；
- Learned-source ablation；
- PF-ODE attribution baseline；
- NFE and wall-clock comparison；
- Mechanism diagnostics。

### 10.5 Discussion

- 为什么 learned source 是核心；
- 哪些结论已由当前实验支持；
- 哪些仍依赖正式服务器实验；
- 方法边界和未来扩展。

## 11. 当前一句话总结

S3FM 的核心不是“把 diffusion 换成 flow matching”，而是利用 flow matching 允许非高斯、信息化、观测相关 source 的自由度，让重建从一个接近目标的低带宽场出发；在保持算子无关 guidance 的同时，用更短传输路径实现少步时空动力学重建。
