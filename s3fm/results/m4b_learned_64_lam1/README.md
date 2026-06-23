# M4b three-way ablation

Info inference source: `learned`.
Eval split: `test`, windows: `64`, seeds: `0`.
Guidance lambda: `1.0`, reduction: `sum`, observed fraction: `0.15`.

## Reconstruction summary

| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |
|---:|---:|---:|---:|:---:|:---:|
| 10 | 0.4824 | 0.0906 | 0.0942 | True | True |
| 20 | 0.4620 | 0.0912 | 0.0947 | True | True |
| 50 | 0.4507 | 0.0916 | 0.0951 | True | True |
| 100 | 0.4471 | 0.0917 | 0.0953 | True | True |

## Mechanism diagnostics

- Train-coupled displacement, Gauss: `49.5046`
- Train-coupled displacement, Info: `2.1703`
- Train-coupled curvature proxy, Gauss: `0.0311`
- Train-coupled curvature proxy, Info: `0.0195`
- Marginal inference source displacement to current eval target: `46.5664`
- Warm inference source displacement to current eval target: `2.6268`
- Learned inference source displacement to current eval target: `3.2505`
- Learned inference source nRMSE to current eval target: `0.1121`

The marginal source is distribution-matched but not target-short. The warm source is a fixed measurement-informed low-frequency reconstruction from `y`, not an oracle `S(X_true)`.

## Gate

Pass all matched NFE: `True`
