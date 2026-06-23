# M4b three-way ablation

Info inference source: `learned`.
Eval split: `test`, windows: `8`, seeds: `0`.
Guidance lambda: `2.0`, reduction: `sum`, observed fraction: `0.15`.

## Reconstruction summary

| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |
|---:|---:|---:|---:|:---:|:---:|
| 10 | 0.2408 | 0.0509 | 0.0558 | True | True |

## Mechanism diagnostics

- Train-coupled displacement, Gauss: `47.5190`
- Train-coupled displacement, Info: `1.8332`
- Train-coupled curvature proxy, Gauss: `0.0304`
- Train-coupled curvature proxy, Info: `0.0193`
- Marginal inference source displacement to current eval target: `42.5471`
- Warm inference source displacement to current eval target: `1.9362`
- Learned inference source displacement to current eval target: `2.8071`
- Learned inference source nRMSE to current eval target: `0.0902`

The marginal source is distribution-matched but not target-short. The warm source is a fixed measurement-informed low-frequency reconstruction from `y`, not an oracle `S(X_true)`.

## Gate

Pass all matched NFE: `True`
