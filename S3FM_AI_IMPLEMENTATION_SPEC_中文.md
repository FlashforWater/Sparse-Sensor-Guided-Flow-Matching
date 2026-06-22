# S3FM:稀疏传感器引导的 Flow Matching（流匹配）

## 可供 AI 执行的研究与实现规范

> **翻译说明：** 本文件是英文权威规范 [`S3FM_AI_IMPLEMENTATION_SPEC.md`](./S3FM_AI_IMPLEMENTATION_SPEC.md) 的完整中文译本。**英文版仍是唯一权威（single source of truth）**；若中英两版冲突，以英文版为准，并须同步更新本译本。代码、符号、变量名、公式、YAML 一律保持原文。

**头号贡献（为何这不只是"换采样器"）：** S3FM 从一个**有信息量的、非高斯的源分布（informative, non-Gaussian source distribution）**（一个廉价的粗化/插值场）出发，传输（transport）到干净的完整场，并在**对依赖耦合（dependent couplings）也成立的通用 flow-matching 引导**下，从稀疏测量重建时空动力学。由此带来的**短传输路径**正是能做到少步数（≤50 NFE）重建的原因；而算子无关（operator-agnostic）的引导，则保留了 S3GM 跨各类测量算子 `H` 的零样本（zero-shot）灵活性。

**首要目标：** 证明一个"信息化源引导的 flow-matching ODE"，能在比 S3GM 少一个数量级的函数评估次数下，从稀疏、含噪、不完整或非线性的测量中重建完整时空场；**并且**该加速可归因于本方法本身，而非仅仅来自把 SDE 换成 ODE 求解器。

**首个必须达成的结果：** 在 KSE 基准上，用至多 50 次 ODE 函数评估接近 1,000 步 S3GM 采样器的重建质量，同时在相同 NFE 预算下击败一个匹配的"S3GM 的 probability-flow ODE 版本"基线。

> 框定警告：在**标准高斯源 + 线性仿射路径 + 独立耦合（standard-Gaussian source + linear affine path + independent coupling）**下，flow matching 在数学上等价于一个仅差 noise schedule 的 diffusion model（见 *On the Guidance of Flow Matching* 及 §18 决策日志）。因此该配置是一个**受控基线（controlled baseline，S3FM-Gauss）**，而非贡献本身。贡献存在于信息化源 / 依赖耦合的范畴（§6.0）。

---

## 1. 给任何使用本文档的 AI 智能体的说明

在修改代码前，智能体必须：

1. 完整阅读本文档。
2. 检查现有仓库，并保留与任务无关的用户改动。
3. 在第 11 节中确认当前里程碑。
4. 仅实现该里程碑，除非用户明确扩大范围。
5. 为引入的每一个数学算子添加或更新测试。
6. 在进行昂贵实验前，先运行最小的相关测试。
7. 报告确切的命令、配置、指标、运行时间，以及尚未解决的风险。

智能体不得：

- 在测量之前就声称有加速或精度提升；
- 悄悄改变数据划分、归一化、网格、物理时间步长或指标定义；
- 混淆物理时间（physical time）与流积分时间（flow integration time）；
- 在 `(测量, 完整场)` 配对上训练条件模型，除非明确要求；
- 把一个新的动力系统当作零样本泛化（zero-shot generalization）；
- 把并行窗口方程与自回归方程合并为同一次更新；
- 在没有单独论证的转换方法的情况下，把旧的 diffusion 权重当作 flow-matching 权重使用。
- **此外：** 在缺少"S3GM 的 probability-flow ODE 版本"基线（§11 M7）时，绝不将实测加速归因于 "flow matching"；绝不把标准高斯源配置当作贡献（它是 `S3FM-Gauss` 受控基线，§6.0）；推理时绝不使用与先验训练所传输的源不同的源分布（§6.0 训练/推理源一致性）。

---

## 2. 研究问题与假设

### 2.1 研究问题

一个从**有信息量的源分布**出发传输的预训练 flow-matching 先验，能否在不为每个测量算子 `H` 重新训练的前提下，从稀疏、含噪、不完整或非线性的测量中，显著快于 S3GM 地重建并预测完整时空动力学；**且该速度优势可被证明来自更短的传输路径，而非仅仅来自 ODE-vs-SDE 的积分方式差异**？

### 2.2 主假设

当一个 flow-matching 先验的**源分布携带关于目标的信息**（例如一个粗化、模糊或插值后的场）时，它所需的传输比高斯源 diffusion 短得多、也更直，因而同样的算子无关协方差梯度引导（covariance-gradient guidance）可以在大约 20–100 次 ODE 函数评估内（而非 1,000 步 SDE）达到 S3GM 级别的重建质量。关键在于，该优势应当**在相同 NFE 下也压过匹配的"S3GM 的 probability-flow ODE 版本"基线**，从而把"信息化源"的贡献与"确定性积分"的贡献分离开。

### 2.3 次级假设

1. 信息化源先验（S3FM-Info）在给定重建质量下，所需 NFE 严格少于高斯源先验（S3FM-Gauss），且 NFE 预算越小、差距越大。
2. 对**依赖耦合**也成立的通用 flow-matching 引导（反向耦合比 `P`，见 *On the Guidance of Flow Matching* §3.1）允许信息化源与目标以非独立方式配对（例如每个粗场与其自身的精场配对），在所声明的 `P ≈ 1` 范围内不破坏引导的正确性。
3. 并行重叠窗口引导可以生成比训练窗口更长的序列，且无可见的时间接缝（temporal seams）。
4. 自回归端点引导可以把预测延伸到观测区间之外。
5. 在计入穿过速度网络的梯度计算后，速度优势仍然有意义。
6. 即使混沌的逐点轨迹发散，统计特征仍可能保持准确。

### 2.4 第一版论文的非目标

- 跨无关动力系统的通用基础模型。
- 对控制方程（governing PDEs）的精确强制。
- 对任意非线性 `H` 的后验精确性证明，或处理 `P ≠ 1` 依赖耦合的新证明（我们工作在 `P ≈ 1` 范围内并引用既有框架）。
- 在正确的 KSE 实现存在之前的实时性能。
- 在 KSE 试点成功之前复现 S3GM 的每一个数据集。

---

## 3. 源论文

1. **S3GM：** *Learning spatiotemporal dynamics with a pretrained generative model*。
   - 可复用思路：无条件时空先验、观测一致性（observation consistency）、序列一致性（sequence consistency）、并行重叠窗口、自回归续推（autoregressive continuation）、由重复采样得到的不确定性。
   - 本项目针对的主要局限：1,000 步引导 SDE 采样。

2. **流引导：** *On the Guidance of Flow Matching*。
   - 可复用思路：对**任意源分布与依赖耦合**都成立的通用引导向量场（其 Eq. 1）、端点估计、`g_cov-G`、`g_cov-A`，以及逆问题专用引导。
   - **对本项目的关键支撑结论：** 引导向量场带有一个反向耦合比 `P = π'(x0|x1)/π(x0|x1)`；在独立耦合下 `P = 1` 精确成立，对 mini-batch-OT 式的依赖耦合也是合理近似。正是它在理论上许可了使用"信息化、依赖耦合的源"（§6.0），同时复用同一套闭式引导。我们工作在 `P ≈ 1` 范围内，不对 `P ≠ 1` 提出新理论。
   - 本项目的初始方法选择：`g_cov-G` 作为主方法，`g_cov-A` 作为面向速度的消融，逆问题专用引导仅用于线性 `H`。

### 3.1 新颖性边界（什么是、什么不是贡献）

把 diffusion 采样器替换为 flow matching，本身**并不**构成方法学贡献——在高斯源加线性路径下，两者仅差一个 schedule。因此一篇站得住的论文至少必须证明：

1. **信息化源时空流先验：** 从一个廉价的结构化场（粗化/模糊/插值）而非高斯噪声出发传输，并在低 NFE 下给出明确、实测的"更短传输"优势。
2. **在（可能依赖的）耦合下的算子无关引导**，用于时空逆问题，复用通用流引导框架。
3. **长序列重建**，通过联合引导的重叠流窗口实现。
4. **能分离因果的速度-精度研究**——在匹配 NFE 下击败"S3GM 的 probability-flow ODE 版本"基线，而不仅仅是 1,000 步 SDE。
5. 至少一项额外贡献，最好是**残差自适应引导（residual-adaptive guidance）或自适应 NFE 分配**。

除非真正发展出新证明，否则不要声称提出了流引导的新通用理论。不要声称跨系统的零样本泛化。

---

## 4. 记号约定（Notation Contract）

以下记号在代码、注释、图表与文档中均为强制约定。

| 符号 | 含义 |
|---|---|
| `t_phys` 或 `n` | 物理时间或物理帧索引 |
| `s` | 流积分时间，从 `0` 增到 `1` |
| `i` | 窗口索引 |
| `X` | 一个干净的完整时空样本 |
| `Z0` | 源样本。**在 S3FM-Gauss 中为高斯噪声；在 S3FM-Info 中为一个信息化结构场（例如 `S(X)`）**（§6.0） |
| `Z1` | 目标干净样本；训练时 `Z1 = X` |
| `S(.)` | 源构造算子，从干净场产生信息化源（例如 粗化-模糊-下采样-上采样）；用于 S3FM-Info |
| `pi(Z0, Z1)` | 耦合（源-目标配对的联合分布）；S3FM-Gauss 中独立，S3FM-Info 中依赖 |
| `Zs` | 流时间 `s` 处的中间流状态 |
| `v_theta(Zs, s)` | 学到的流速度 |
| `X1_hat` | 由当前流状态估计的干净端点 |
| `y` | 来自当前真实或仿真传感器的测量 |
| `H` | 已知的可微测量算子 |
| `T` | 一个模型窗口中的物理帧数 |
| `T_total` | 期望的总物理序列长度 |
| `m` | 相邻窗口间重叠的物理帧数 |
| `T_init` | 约束一个自回归窗口的初始帧数 |
| `NFE` | ODE 采样中速度网络的函数评估次数 |

在实现文档中，绝不可用同一个符号 `t` 同时表示物理时间与流时间。

---

## 5. 问题表述

未知的干净序列为：

```text
X in R^[T_total, C, spatial dimensions]
```

测量遵循：

```text
y = H(X) + measurement_noise
```

`H` 可以表示：

- 稀疏点采样；
- 规则下采样；
- 缺失的物理变量；
- 傅里叶空间测量；
- 时间平均；
- 速度幅值 `sqrt(u^2 + v^2)`；
- 以上的组合。

期望的后验在概念上为：

```text
p(X | y) proportional to p_prior(X) * exp(-J_obs(X))
```

对于高斯测量噪声：

```text
J_obs(X) = 0.5 / sigma_y^2 * ||y - H(X)||_2^2
```

在代码中，默认使用**均方残差（mean squared residual）**而非原始求和，这样引导尺度不会仅因观测数量变化而改变。若复现某篇论文使用求和的公式，则把这个归约方式做成可配置并记录下来。

---

## 6. 方法规范

### 6.0 源分布与耦合设计（头号贡献）

S3FM 定义为两个变体。**两者都必须实现**，因为第二个是贡献，第一个是它的对照。

**S3FM-Gauss（受控基线）。** 标准高斯源、独立耦合、线性仿射路径。它在数学上等价于一个仅差 noise schedule 的 diffusion（§18）。它的作用是：(a) 在最简单的设定上打通全部管线（plumbing）；(b) 量化在每个 NFE 预算下信息化源到底带来多少收益。

```text
Z0 ~ Normal(0, I)        # pure noise, carries no information about X
pi(Z0, Z1) = p(Z0) p(Z1) # independent coupling, so P = 1 exactly
```

**S3FM-Info（贡献本身）。** 源是目标的一个廉价、结构化近似，由一个确定性的、破坏信息的算子 `S` 产生：

```text
Z1 = X ~ p_data
Z0 = S(X) + eta          # eta is small residual noise to keep the source non-degenerate
pi(Z0, Z1)               # dependent coupling: each source is paired with ITS OWN target
```

`S` 是一个固定、廉价的映射，丢弃的恰好是引导必须恢复的信息。针对 KSE 的候选构造，按真实度递增：

1. **粗化-模糊源（coarse-blur source）：** 按与测量算子相同的因子做空间下采样，再上采样回去（例如 模糊 + 抽取 + 插值）。该源就是"对一个通用稀疏布局做朴素插值会得到的东西"。
2. **谱截断源（spectral-truncation source）：** 保留 `X` 最低的 `k` 个傅里叶模态，其余置零。然后由流去合成细尺度结构。
3. **低保真代理源（low-fidelity surrogate source，仅用于后续系统）：** 一次快速的降阶或粗网格求解。预留给 Kolmogorov/ERA5；KSE 试点不需要。

先验所用的源算子 `S` **不得依赖具体的推理时 `H`**——否则算子无关的叙事就会崩塌，模型不过是一个伪装的条件回归器 `(y → X)`。`S` 定义的是一个通用、固定的腐蚀族；推理时的 `y, H` 再在先验内部进行引导。这种分离正是全部要点，必须在论文中明确陈述。

**为何这能带来少步数采样（我们将要实测的机制）。** 在线性路径 `Zs = (1-s)Z0 + sZ1` 下，当 `Z0 = S(X)+eta` 时，每个样本的传输位移为 `Z1 - Z0 = X - S(X) - eta`，即**只有缺失的细节**，而非整张场。位移更小、曲率更低，在给定误差下所需的 Euler/RK 步数就更少。我们将**实测**这一点而非假设它（见 §12 的路径长度/直度诊断，以及 M3/M4b 的 NFE 扫描）。

**耦合的合法性（`P ≈ 1` 范围）。** *On the Guidance of Flow Matching*（Eq. 1）的通用引导带有反向耦合比 `P = π'(x0|x1)/π(x0|x1)`。S3FM-Info 使用**确定性**源映射，因此 `π(x0|x1)` 在 `x0 = S(x1)` 处尖锐峰化；被引导的分布对 `x1` 重新加权，但保持同一条件源映射，从而 `P ≈ 1`。我们采用与源论文对依赖（mini-batch-OT）耦合相同的 `P = 1` 近似，将其作为一个明确假设陈述，并标注任何使其可疑的情形。不声称任何 `P ≠ 1` 的新理论。

**训练/推理源一致性（不可妥协）。** 速度场只在它训练过的路径族上有效。因此：

- 预训练中所用的**同一个** `S`（以及 `eta` 的同一残差噪声尺度）必须用于推理时抽取 `Z0`。
- 推理时我们**没有** `X`，所以推理源必须由测试时可得的量构造，且要匹配训练源的*分布*。可接受的构造（每个实验需选定并冻结）：
  - **边际源采样（marginal-source sampling）：** 从训练源 `S(X)+eta` 的经验/参数化分布中抽取 `Z0`（信息无关；最安全，不破坏任何假设）。
  - **由 y 暖启动（warm-start-from-y，一个刻意的、单独消融的耦合选择）：** 用一个*固定、通用*的重建（例如 先插值再模糊以匹配 `S` 的统计量）把测量映射回场空间来构造 `Z0`。这会把观测信息注入源中。它能进一步缩短传输，但改变了推理耦合，**必须作为一个独立方法（`S3FM-Info-warm`）报告**，并保留边际源变体作为干净对照。
- 必须有一个单元测试断言：推理时 `Z0` 与训练时 `Z0` 的概要统计量（逐通道均值/标准差与能谱）在容差内匹配。

**范围纪律。** S3FM-Info 是头号贡献。S3FM-Gauss 作为对照是必做的。在 S3FM-Gauss 管线通过 M0–M4 之前，不要因 `S` 的设计而阻塞进度；源算子可在单窗口高斯引导跑通后于 M4b（§11）引入。

### 6.1 无条件 flow-matching 预训练

对 S3FM-Gauss，使用独立高斯源和仿射线性路径：

```text
Z0 ~ Normal(0, I)
Z1 = X ~ p_data
s  ~ Uniform(0, 1)
Zs = (1 - s) * Z0 + s * Z1
target_velocity = Z1 - Z0
```

对 S3FM-Info，仅替换源的抽取与耦合（路径、损失、端点公式不变）：

```text
Z1 = X ~ p_data
Z0 = S(X) + eta          # dependent coupling; same S used at inference (§6.0)
s  ~ Uniform(0, 1)
Zs = (1 - s) * Z0 + s * Z1
target_velocity = Z1 - Z0
```

训练（两个变体目标相同）：

```text
L_FM = mean_square(v_theta(Zs, s) - target_velocity)
```

要求的模型输入与输出形状：

```text
input Zs:  [batch, physical_time, channels, *spatial]
input s:   [batch] or broadcast-compatible embedding
output:    exactly the same shape as Zs
```

首个模型应复用 Video U-Net 的时空归纳偏置：

- 空间卷积；
- 时间注意力或时间混合；
- 跳连（skip connections）；
- 连续的流时间嵌入。

预训练先验时不要加入传感器信息。

### 6.2 干净端点估计

对线性路径：

```text
X1_hat = Zs + (1 - s) * v_theta(Zs, s)
```

它扮演 S3GM 中 Tweedie 干净估计的角色。

要求的单元测试：

- 用在精确线性路径上求得的 oracle 速度 `v = Z1 - Z0`，对若干 `s` 值，`X1_hat` 必须在数值精度内等于 `Z1`。

### 6.3 主引导：协方差梯度引导

定义一个正的引导强度 `lambda_s >= 0`，并采用显式的下降约定：

```text
g_cov_G = -lambda_s * grad_Zs J(X1_hat)
guided_velocity = v_theta(Zs, s) + g_cov_G
```

`grad_Zs` 同时穿过端点表达式和 `v_theta` 求导。这需要一次穿过速度网络的向量-雅可比积（vector-Jacobian product）。

参考 autograd 模式（主方法中**不要** detach `velocity` 或 `x1_hat`）：

```python
zs = zs.requires_grad_(True)
velocity = model(zs, flow_time)
x1_hat = zs + (1.0 - flow_time) * velocity        # source-agnostic endpoint (§6.2)
energy = energy_fn(x1_hat)                          # scalar, normalized
grad = torch.autograd.grad(energy, zs, create_graph=False)[0]
guidance = -lambda_s * grad                         # explicit descent; sign is unit-tested
guided_velocity = velocity + guidance
```

重要实现规则：

- 不要凭记忆或凭一个朝向不同的时间约定去推断引导符号。
- 局部引导单元测试必须验证：当基础速度被禁用时，一个足够小的 Euler 步会降低 `J`。
- 端点估计 `X1_hat = Zs + (1-s) v_theta` 仅由线性路径导出，因此对 **S3FM-Gauss 与 S3FM-Info 是相同的**；无需任何与源相关的端点代码。

### 6.4 快速消融：端点梯度引导

使用：

```text
g_cov_A = -lambda_s * grad_X1_hat J(X1_hat)
```

把该梯度当作状态空间中的一个向量，不穿过 `v_theta` 求导。

它更便宜但更近似。它是一个消融，不是最初的主张。

### 6.5 依赖观测的长序列重建

把观测序列拆成 `B` 个长度为 `T`、重叠 `m`、步幅为 `T - m` 的窗口。

`T = 5`、`m = 2` 的例子：

```text
window 1: frames 1  2  3  4  5
window 2: frames          4  5  6  7  8
window 3: frames                   7  8  9 10 11
```

对窗口 `i`，计算 `X1_hat[i]`。拼装出一个用于测量评估的全局估计序列。

观测能量：

```text
J_obs = mean_square(y - H(global_X1_hat)) / (2 * sigma_y^2)
```

序列一致性能量：

```text
J_seq = mean over adjacent windows and overlap entries of
        ||overlap(X1_hat[i + 1]) - stop_grad(overlap(X1_hat[i]))||^2
```

并行观测阶段（方程 A）的组合能量：

```text
J_parallel = alpha * J_obs + beta * J_seq
```

所有窗口作为同一个联合 ODE 状态一起积分。本阶段使用 `alpha` 与 `beta`；不使用 `gamma`。

### 6.6 独立于观测的自回归预测

测量结束后，一次预测一个窗口。

给定上一个已完成序列的最后 `T_init` 帧：

```text
J_init = mean_square(
    first_T_init_frames(X1_hat),
    given_initial_frames
)
```

自回归阶段（方程 B）的能量：

```text
J_AR = gamma * J_init
```

生成该窗口，仅追加其新帧，再用其最后 `T_init` 帧作为下一个窗口的条件。

本阶段使用 `gamma`；不使用 `alpha`、`beta` 或测量 `y`，除非实验明确执行在线数据同化（online data assimilation）。

### 6.7 不确定性估计

对固定的初始源样本，ODE 是确定性的，但可通过抽取不同的 `Z0` 来保留不确定性：

```text
Z0^(k) ~ Normal(0, I)
X_hat^(k) = GuidedFlow(Z0^(k), y, H)
```

报告跨独立源样本的均值预测与标准差。

---

## 7. 测量算子接口

每个算子都必须提供元数据与确定性行为。

推荐接口：

```python
class MeasurementOperator:
    def forward(self, full_state):
        """Map full clean state to measurement-shaped tensor."""

    def metadata(self):
        """Return locations, physical times, variables, units, and normalization."""

    @property
    def is_linear(self):
        return False
```

线性算子可选地实现：

```python
def adjoint(self, residual):
    """Apply H^T for tests or inverse-specific guidance."""
```

初始算子，按要求顺序：

1. `IdentityOperator`，用于调试。
2. `MaskOperator`，用于随机稀疏 KSE 观测。
3. `RegularDownsampleOperator`。
4. `FourierSubsampleOperator`。
5. `TemporalAverageOperator`。
6. `VelocityMagnitudeOperator`，用于非线性圆柱测量。

每个 `y` 都必须携带或引用产生它所用的确切 `H` 元数据。一个没有传感器位置、物理时间、观测变量与测量类型的纯数值张量，是无效输入。

---

## 8. 引导强度与数值稳定性

先用一个恒定的正 `lambda_s = lambda0` 调试。然后比较各种 schedule。

要求消融的 schedule：

1. 恒定。
2. 论文启发的、朝 `s = 1` 衰减。
3. 一个有界的残差自适应 schedule，仅在恒定基线跑通后引入。

安全机制必须可配置且被记录：

- 梯度范数裁剪；
- 引导与先验速度之比的上限；
- `lambda_s` 的最小与最大值；
- ODE 求解器容差；
- 定步长 vs 自适应步长积分。

在每个被采样的流时间，始终记录：

```text
||v_theta||
||g||
J_obs
J_seq or J_init
measurement residual
```

不要靠只报告成功的种子来掩盖不稳定的运行。

### 8.1 NFE 计数（每个速度主张都强制）

NFE 计的是**每一次对 `v_theta` 的调用**，而非求解器步数。先从定步长 Euler 求解器开始，因为它让符号、梯度与 NFE 都透明；之后再加定步长 midpoint/RK4。不要一开始就用自适应黑盒求解器——它会隐藏 NFE 并使推理时的 autograd 复杂化。

| 求解器 | 步数 | NFE（前向） |
|---|---|---|
| Euler | 50 | 50 |
| Midpoint | 50 | 100 |
| RK4 | 50 | 200 |

绝不要把"50 步 RK4"与"50 步 Euler"当作模型成本相等来比较。同时报告名义步数与真实 NFE。

**`g_cov-G` 的成本。** 每次被引导的 NFE 还要额外穿过 `v_theta` 反向传播（一次 VJP），因此其每次评估的墙钟成本约为无引导前向调用的 2–3 倍。`g_cov-A` 不穿过 `v_theta` 反传，成本约 1 倍再加一个端点梯度。运行时表（§20）必须反映这一点；一次"NFE 匹配"的比较**并不**自动等于墙钟匹配的比较，两者都必须报告。

---

## 9. 推荐的代码组织

```text
src/
  data/
    kse.py
    normalization.py
  models/
    video_unet.py
    time_embedding.py
  flow/
    sources.py        # S(.) source operators + inference-source samplers (S3FM-Info, §6.0)
    paths.py
    losses.py
    endpoint.py
    ode_sampler.py
  guidance/
    energies.py
    cov_g.py
    cov_a.py
    schedules.py
  measurements/
    base.py
    identity.py
    mask.py
    downsample.py
    fourier.py
    temporal.py
    nonlinear.py
  sequences/
    windows.py
    assemble.py
    autoregressive.py
  metrics/
    reconstruction.py
    statistics.py
    runtime.py
  train_flow.py
  sample_guided.py
  evaluate.py
configs/
  kse_flow.yaml
  kse_guided_mask.yaml
tests/
  test_flow_path.py
  test_endpoint.py
  test_measurements.py
  test_guidance_direction.py
  test_windows.py
  test_sampler_smoke.py
experiments/
  README.md
  results.csv
```

请把该结构适配到现有仓库，而不要重复已建立的抽象。

---

## 10. 最小配置字段

每个实验都必须保存一份已解析（resolved）的配置，包含：

```yaml
seed: 0
dataset:
  name: kse
  train_split: null
  val_split: null
  test_split: null
  physical_dt: null
  spatial_resolution: null
  window_length: null
  normalization: null

flow:
  source: standard_gaussian
  path: linear
  model: video_unet
  checkpoint: null

sampler:
  solver: euler_or_selected_solver
  nfe: 50
  adaptive: false

guidance:
  method: cov_g
  lambda0: null
  schedule: constant
  gradient_clip: null
  max_guidance_prior_ratio: null

measurement:
  operator: mask
  observed_fraction: null
  noise_std: null

sequence:
  overlap: null
  initial_frames: null

evaluation:
  num_samples: 3
  metrics:
    - nrmse
    - observation_residual
    - two_point_correlation
    - runtime
```

没有已解析配置与模型 checkpoint 标识，任何实验结果都无效。

---

## 11. 里程碑与验收标准

### M0. 仓库与可复现性基础

- [ ] 检查仓库与依赖环境。
- [ ] 加入确定性种子处理。
- [ ] 加入配置加载与已解析配置保存。
- [ ] 加入实验结果到 CSV 或 JSONL 的记录。
- [ ] 确认 CPU 冒烟测试（smoke test）可在无 GPU 下运行。

**验收：** 两次相同种子的相同冒烟运行，在数值容差内产生相同的损失与样本。

### M1. KSE 数据管线

- [ ] 加载或生成 KSE 轨迹。
- [ ] 冻结 训练/验证/测试 划分。
- [ ] 实现可逆归一化。
- [ ] 构造恰好 `T` 个物理帧的窗口。
- [ ] 可视化若干干净样本并核对坐标轴。

**验收：** 归一化往返误差低于 `1e-6`；无轨迹跨划分泄漏；样本维度有文档记录。

### M2. 无条件 flow-matching 先验

- [ ] 实现线性条件路径。
- [ ] 实现 flow-matching 损失。
- [ ] 在一个极小 batch 上训练一个小的过拟合模型。
- [ ] 训练首个完整 KSE 先验。
- [ ] 保存 checkpoint 与训练曲线。

**验收：** 极小 batch 损失大幅下降；端点 oracle 测试通过；无引导样本有限且非常数。

### M3. 无引导先验评估

- [ ] 以 NFE `10, 20, 50, 100` 采样。
- [ ] 把边际取值分布与两点相关（two-point correlation）和测试数据比较。
- [ ] 测量墙钟运行时间与峰值显存。

**验收：** 在加入测量引导之前，至少有一个 NFE 设置能产生统计上可信的 KSE 样本。

### M4. 单窗口稀疏观测引导

- [ ] 实现 `IdentityOperator` 与 `MaskOperator`。
- [ ] 实现 `J_obs`。
- [ ] 实现 `g_cov-G`。
- [ ] 加入引导方向单元测试。
- [ ] 从已知稀疏观测重建一个 KSE 窗口。
- [ ] 扫一小组 `lambda0` 值。

**验收：** 在相同观测与种子下，被引导的重建比无引导采样有更低的测量残差与更低的平均 nRMSE。

> M2–M4 使用 **S3FM-Gauss** 先验（高斯源）。这是在引入 M4b 的信息化源之前，先在 diffusion 等价的对照上打通全部管线。

### M4b. 信息化源流先验 —— 头号贡献

这是核心的科学里程碑，不是可选的扩展。在 M4 管线通过后做。

- [ ] 在 `flow/sources.py` 中实现 `S(.)` 源算子（KSE 先从 粗化-模糊 与 谱截断 开始）。
- [ ] 实现匹配的推理源采样器（先做边际源变体；`warm-start-from-y` 仅作为单独标注的第二种方法）。
- [ ] 加入 源确定性、训练/推理源一致性、传输缩短 测试（§13.13–13.15）。
- [ ] 用依赖耦合 `Z0 = S(X)+eta` 训练 **S3FM-Info** 先验（损失、路径、端点与 S3FM-Gauss 相同）。
- [ ] 用信息化先验重跑 M4 的单窗口引导重建。
- [ ] 测量两种先验的平均传输位移 `||Z1 - Z0||` 与路径曲率。

**验收：** (a) 信息化源的端点与一致性测试通过；(b) 在匹配 NFE 下，S3FM-Info 达到比 S3FM-Gauss 更低（或相等）的重建误差，且 NFE 越小差距越大；(c) 实测的传输位移/曲率对信息化源更小，佐证该机制。若 (b) 失败，如实报告——这会证伪主假设，必须暴露而非掩埋。

### M5. 并行长序列重建

- [ ] 实现重叠窗口分解。
- [ ] 实现 stop-gradient 序列一致性。
- [ ] 联合积分所有观测窗口。
- [ ] 拼装最终长序列。
- [ ] 比较有无 `J_seq` 的情况。

**验收：** 序列一致性在不明显增大观测残差的情况下降低重叠不一致。无重复或缺失的物理帧索引。

### M6. 自回归未来预测

- [ ] 实现 `J_init`。
- [ ] 顺序生成未来窗口。
- [ ] 仅追加新帧。
- [ ] 记录逐点误差与统计误差随预测时域的变化。

**验收：** 首批生成帧与给定的初始帧匹配；长时域输出保持有限；误差累积被绘出而非隐藏。

### M7. 主速度-质量基准

- [ ] 比较 1,000 步的 S3GM 与 NFE `10, 20, 50, 100` 的 S3FM。
- [ ] **归因基线（强制）：** 把*同一个 S3GM score model* 作为 probability-flow ODE、配*相同引导*在 NFE `10, 20, 50, 100` 积分（`S3GM-PF-ODE`）。这把"ODE vs SDE"与"S3FM 方法本身"分离开。
- [ ] 在每个 NFE 都报告 S3FM-Gauss 与 S3FM-Info，使信息化源的收益相对 diffusion 等价对照可见。
- [ ] 使用相同的测试用例、观测、归一化与硬件。
- [ ] 在 S3FM 运行时计入梯度计算时间；报告 NFE 匹配**与**墙钟匹配两种比较（§8.1）。
- [ ] 至少重复三个种子。

**首要成功标准：** 在 NFE ≤ 50 时，S3FM-Info 达到接近 S3GM@1,000 的预先声明质量阈值，**且在相同 NFE 下击败 `S3GM-PF-ODE`**，并有有意义的实测墙钟加速。若 S3GM-PF-ODE 在低 NFE 下已能与 S3FM 持平，则诚实的结论是收益来自确定性积分而非本方法——若出现这种情况，要明白地陈述出来。

确切的质量阈值必须在最终基准之前固定。一个合理的试点阈值是 nRMSE 与统计特征误差相对退化不超过 10%，但在复现 S3GM 基线后必须重新审视。

### M8. 引导与源消融

- [ ] `g_cov-G` vs `g_cov-A`。
- [ ] **源分布：** 高斯 vs 粗化-模糊 vs 谱截断 vs warm-start-from-y，在匹配 NFE 下。
- [ ] **耦合敏感性：** 改变残差噪声尺度 `eta`（它控制耦合的依赖程度），检查引导稳定性 / `P ≈ 1` 假设。
- [ ] 恒定 vs 带 schedule 的引导。
- [ ] 不同观测稀疏度与噪声。
- [ ] 不同重叠大小。
- [ ] 定步长 vs 自适应 ODE 求解器。

**验收：** 每次消融恰好改变一个主因子，并使用相同的评估集。

### M9. 更多系统

仅在 M7 成功后进行。

推荐顺序：

1. Kolmogorov 流。
2. ERA5 不完整变量重建。
3. 带非线性速度幅值观测的圆柱绕流。

每个新系统都需要它自己的无条件先验。不要把这描述为跨系统的零样本迁移。

---

## 12. 实验矩阵

最小 KSE 矩阵：

| 因子 | 取值 |
|---|---|
| 方法 | S3GM（1,000 步 SDE）、**S3GM-PF-ODE（归因基线）**、无引导 FM、`g_cov-A`、`g_cov-G` |
| 源 / 耦合 | **高斯（S3FM-Gauss 对照）、粗化-模糊、谱截断、warm-start-from-y** |
| NFE | 10, 20, 50, 100；S3GM 参照 1,000；S3GM-PF-ODE 在 10, 20, 50, 100 |
| 观测 | 随机掩码、规则下采样、傅里叶子采样、初始帧 |
| 稀疏度/下采样 | 至少 3 个级别，含一个困难级别 |
| 噪声 | 0 与至少一个非零级别 |
| 序列一致性 | 关 / 开 |
| 种子 | 至少 3 个 |

要求的指标：

- 归一化 RMSE（normalized RMSE）；
- 测量空间中的观测残差；
- KSE 的空间两点相关误差；
- 重叠不一致（overlap disagreement）；
- 墙钟采样时间；
- NFE；
- 峰值显存（若可得）；
- 跨种子的均值与标准差。
- **平均传输位移 `||Z1 - Z0||` 与离散 ODE 路径曲率**（用于佐证更短传输的机制并比较各源）；
- 针对 S3FM-Info 的**训练/推理源分布漂移诊断**（训练源与推理源之间的逐通道均值/标准差与谱距离）。

对湍流，额外报告动能谱（kinetic energy spectrum）误差。

---

## 13. 强制单元与集成测试

1. **线性路径测试：** `Zs` 等于解析插值（两种源都测）。
2. **速度目标测试：** 解析导数等于 `Z1 - Z0`（两种源都测）。
3. **端点测试：** oracle 速度恢复 `Z1`，对高斯**与**信息化 `Z0` 都验证（端点公式与源无关）。
4. **恒等测量测试：** `H(X) = X` 精确成立。
5. **掩码测量测试：** 观测条目与形状精确无误。
6. **线性 H 的伴随测试：** `<Hx, y> = <x, H^T y>` 在容差内成立。
7. **引导符号测试：** 一个小的仅引导步会降低能量。
8. **零引导等价：** `lambda0 = 0` 与无引导 ODE 输出一致。
9. **窗口索引测试：** 分解与拼装保留每一个全局帧。
10. **重叠损失测试：** 相同重叠给零损失；扰动后的重叠给正损失。
11. **自回归追加测试：** 去除重叠后无重复输出帧。
12. **梯度有限性测试：** 对代表性输入，所有引导梯度都有限。
13. **源确定性测试（S3FM-Info）：** 给定 `eta` 的种子，`S(X)` 是确定性的，且重复施加 `S` 足够幂等以可复现。
14. **训练/推理源一致性测试（S3FM-Info）：** 推理时 `Z0` 与训练时 `S(X)+eta` 在逐通道均值/标准差与能谱上于声明容差内匹配；对刻意不匹配的源，该测试必须高调失败。
15. **传输缩短合理性测试：** 在相同目标上，信息化源的平均 `||Z1 - Z0||` 与离散路径曲率严格小于高斯源。

---

## 14. 常见失败模式

### 数学失败

- 因颠倒 diffusion 与 flow 时间约定而导致的引导符号错误。
- 把 `H` 作用于 `Zs` 而非估计的干净端点。
- 在不同归一化空间中比较测量与生成值。
- 把物理帧索引当作流时间。
- 用着 `g_cov-A` 却声称是 `g_cov-G`。
- 反向传播进入 stop-gradient 重叠目标的两侧。
- 让源算子 `S` 依赖推理 `H`，把"先验"变成伪装的条件回归器。
- 在确定性源耦合使 `P ≠ 1` 时却假设引导是精确的。

### 数值失败

- 引导范数压过先验速度。
- 端点估计在靠近源处很差，导致早期引导不稳。
- 强引导把 ODE 路径掰弯，抹掉了预期的 NFE 优势。
- 自适应求解器执行许多隐藏评估；报告真实 NFE，而非名义步数。
- 自回归预测把累积误差当作真值回灌。
- 推理源抽自与训练源不同的分布，导致在 `s = 0` 处对 `v_theta` 的查询离开了训练路径。

### 科学失败

- 用不同预处理或数据划分比较各方法。
- **在没有 `S3GM-PF-ODE` 基线时把加速归因于 "flow matching"**（它可能只是 ODE-vs-SDE）。
- **在没有匹配 NFE 的高斯源消融与实测传输缩短时，就声称信息化源有帮助。**
- 只报告测量残差而忽略全场误差。
- 对混沌系统只报告 nRMSE 而忽略统计特征。
- 把一个貌似合理的未观测变量称作"测量到的恢复"。
- 仅凭理论步数缩减就声称实时性能。
- 声称跨系统的零样本泛化。

---

## 15. AI 任务提示模板

每次实现会话使用此模板：

```text
Read docs/S3FM_AI_IMPLEMENTATION_SPEC.md completely.

Current milestone: M[number and name].

Your task:
[one bounded task]

Required behavior:
1. Inspect existing code before editing.
2. Preserve unrelated changes.
3. State the exact mathematical convention you will implement.
4. Add or update focused tests.
5. Run the smallest relevant test suite.
6. Do not start later milestones.
7. Report changed files, tests, results, assumptions, and remaining risks.

Acceptance criteria:
[copy the relevant criteria from Section 11]
```

M4 的示例：

```text
Read docs/S3FM_AI_IMPLEMENTATION_SPEC.md completely.

Current milestone: M4, single-window sparse observation guidance.

Implement MaskOperator, normalized Gaussian observation energy, and g_cov-G for one KSE window. Use the explicit convention g = -lambda * grad_Zs J(X1_hat). Add tests proving that the mask is correct, lambda=0 reproduces unguided sampling, gradients are finite, and a small guidance-only step reduces J. Do not implement overlapping windows or autoregressive generation yet.
```

---

## 16. AI 完成报告模板

每次 AI 实现回复都应以以下内容结尾：

```text
Milestone:

Outcome:

Files changed:

Mathematical conventions:

Tests executed and results:

Experiment configuration:

Measured metrics/runtime:

Known limitations or risks:

Next permitted step:
```

---

## 17. 论文结构

1. **引言（Introduction）**
   - 稀疏传感器重建很重要。
   - S3GM 灵活但慢（1,000 步 SDE）。
   - Flow matching 使从信息化源出发的传输成为可能，但从此类源做时空后验引导尚未确立；高斯源流不过是重新 schedule 的 diffusion，因此源才是杠杆所在。

2. **相关工作（Related Work）**
   - 时空重建。
   - 基于 score 的逆问题与 S3GM。
   - Flow matching、通用流引导与耦合。
   - 科学生成建模。

3. **问题设定（Problem Setup）**
   - `y = H(X) + noise`。
   - 零样本指的是：对一个固定动力系统的新测量算子。

4. **方法（Method）**
   - 信息化源时空流先验及其依赖耦合（`P ≈ 1` 范围）。
   - 用于通用 flow matching 的端点观测引导。
   - 并行序列一致性。
   - 自回归续推。
   - 求解器、NFE 计数与更短传输机制。

5. **实验（Experiments）**
   - 精度、统计、不确定性、运行时间、NFE。
   - 线性与非线性测量。
   - 消融。

6. **局限（Limitations）**
   - 近似引导。
   - 先验漂移（prior shift）。
   - 强引导路径曲率。
   - 自回归误差累积。
   - 逐系统预训练。

7. **结论（Conclusion）**

在 M7 完成之前，不要把任何数值主张写进摘要。

---

## 18. 决策日志

在此记录所有有后果的决策。

| 日期 | 决策 | 理由 | 状态 |
|---|---|---|---|
| 2026-06-22 | 首版实现采用独立高斯源与线性仿射路径 | 最简单的设定；契合 `g_cov-G` 假设与干净端点公式 | 作为*贡献*已被取代（现为 S3FM-Gauss 对照）；保留为管线基线 |
| 2026-06-22 | 采用 `g_cov-G` 作为主方法 | 支持可微非线性 `H`，且最接近 S3GM/DPS 引导 | 生效 |
| 2026-06-22 | 仅把 `g_cov-A` 用作初始速度消融 | 更便宜但更近似 | 生效 |
| 2026-06-22 | 从 KSE 开始 | 成本最低、兼具逐点与统计指标的受控基准 | 生效 |
| 2026-06-22 | 把 <= 50 NFE 作为首个性能目标 | 相对 1,000 步 S3GM 给出一个具体的速度目标 | 暂定 |
| 2026-06-22 | **把信息化源 / 依赖耦合流（S3FM-Info）提升为头号贡献；把高斯源降为受控基线** | 高斯源线性路径在 schedule 意义下等价于 diffusion，无法承载新颖性；信息化源缩短传输（真正的少步数机制）并利用了 flow matching 独有的非高斯源能力 | 生效 |
| 2026-06-22 | **加入 `S3GM-PF-ODE`（S3GM score model 在 probability-flow ODE 下 + 相同引导）作为强制归因基线** | 没有它，任何加速都会被错误归因于 "flow matching"，而它可能源自 ODE-vs-SDE 积分 | 生效 |
| 2026-06-22 | **工作在 `P ≈ 1` 反向耦合比范围内；把 `P ≠ 1` 修正视为范围外** | 源论文的引导假设 `P ≈ 1`；确定性源映射让我们留在此范围内；推导 `P ≠ 1` 修正将是我们此处不主张的另一项理论贡献 | 生效 |
| 2026-06-22 | **整合规范：本文件为唯一权威；`S3FM_CORE_SPEC.md` 变为指针存根；`S3FM_START_HERE.md` 与 checklist 指向本文件** | 此前两份规范都自称权威，存在漂移风险 | 生效 |

---

## 19. 项目成功的定义

仅当实验证明以下全部时，项目才算成功：

1. 流先验生成统计上可信的无条件动力学。
2. 被引导的样本匹配稀疏测量，并相对无引导样本改善全场重建。
3. **信息化源先验（S3FM-Info）在匹配 NFE 下击败高斯源对照（S3FM-Gauss），且有实测的传输位移/曲率下降来解释该收益。**
4. **在 NFE ≤ 50 时，S3FM 在相同 NFE 下击败 `S3GM-PF-ODE` 归因基线——即该优势不仅仅是 SDE→ODE。**
5. 长序列不产生不可接受的重叠接缝。
6. 自回归预测保持数值稳定，并诚实报告误差累积。
7. 计入引导梯度后的真实墙钟运行时间，相对 S3GM 有有意义的改善。
8. 精度-速度权衡在多个 NFE 上报告，而非只在一个有利设置上。
9. 主张仅限于所测试的动力系统与测量条件。

期望的头号成果是：

> 通过从信息化源而非高斯噪声出发传输，一个免训练、测量算子灵活的引导流，用少一个数量级的函数评估就能以接近 S3GM 的精度重建时空动力学，并有经过验证的墙钟加速——且该优势在匹配 NFE 下也压过 S3GM 的 probability-flow ODE 版本，从而确认收益来自方法本身，而非仅仅来自确定性积分。
