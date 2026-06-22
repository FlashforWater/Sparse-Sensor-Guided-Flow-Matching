# M4 — single-window sparse guided reconstruction (S3FM-Gauss)

g_cov-G guidance reconstructs a KSE window from 15% random sparse observations.

Gate (PASS): guidance monotonically lowers observation residual AND full-field
nRMSE, far below the unguided baseline.

| lambda | nRMSE | obs_resid |
|-------:|------:|----------:|
| 0 (unguided) | 1.284 | 1.284 |
| 1.0 | 0.502 | 0.462 |
| 2.0 | 0.237 | 0.207 |
| 5.0 | 0.083 | 0.070 |
| 10.0 | 0.044 | 0.035 |

NFE sweep (lambda=5): nRMSE ~0.081 at NFE=10, 20, 50, 100 — already strong at
10 NFE (early sign of the few-step reconstruction goal).

Correctness verified: guidance-sign test (a small guided step lowers J),
no-guidance equivalence (lambda=0 == unguided), finite gradients, gradient flows
through v_theta (g_cov-G, not g_cov-A).

Scale note: J_obs uses "mean" reduction by default (scale-independent of #obs),
which dilutes the gradient by N=T*Nx=1280; sweep lambda accordingly, or use
"sum" reduction (used in this sweep) for un-diluted gradients. Both give
identical descent direction. This addresses the spec's "guidance too weak" risk.
