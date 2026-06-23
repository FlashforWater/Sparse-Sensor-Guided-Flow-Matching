# M4b three-way ablation

Info inference source: `learned`.
Eval split: `test`, windows: `64`, seeds: `0,1,2`.
Guidance lambda: `2.0`, reduction: `sum`, observed fraction: `0.15`.

## Reconstruction summary

| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |
|---:|---:|---:|---:|:---:|:---:|
| 10 | 0.2816 | 0.0967 | 0.1003 | True | True |
| 20 | 0.2707 | 0.0976 | 0.1008 | True | True |
| 50 | 0.2660 | 0.0983 | 0.1012 | True | True |
| 100 | 0.2646 | 0.0986 | 0.1013 | True | True |

## Mechanism diagnostics

- Train-coupled displacement, Gauss: `49.5727`
- Train-coupled displacement, Info: `2.1694`
- Train-coupled curvature proxy, Gauss: `0.0311`
- Train-coupled curvature proxy, Info: `0.0195`
- Marginal inference source displacement to current eval target: `47.8945`
- Warm inference source displacement to current eval target: `2.7434`
- Learned inference source displacement to current eval target: `3.4588`
- Learned inference source nRMSE to current eval target: `0.1175`

The marginal source is distribution-matched but not target-short. The warm source is a fixed measurement-informed low-frequency reconstruction from `y`, not an oracle `S(X_true)`.

## Gate

Pass all matched NFE: `True`
