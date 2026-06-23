# M4b three-way ablation

Info inference source: `learned`.
Eval split: `test`, windows: `64`, seeds: `0`.
Guidance lambda: `2.0`, reduction: `sum`, observed fraction: `0.15`.

## Reconstruction summary

| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |
|---:|---:|---:|---:|:---:|:---:|
| 10 | 0.2466 | 0.0896 | 0.0942 | True | True |
| 20 | 0.2365 | 0.0905 | 0.0947 | True | True |
| 50 | 0.2321 | 0.0912 | 0.0951 | True | True |
| 100 | 0.2307 | 0.0915 | 0.0953 | True | True |

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
