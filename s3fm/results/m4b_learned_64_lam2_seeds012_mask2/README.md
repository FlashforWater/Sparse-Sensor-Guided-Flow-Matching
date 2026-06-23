# M4b three-way ablation

Info inference source: `learned`.
Eval split: `test`, windows: `64`, seeds: `0,1,2`.
Guidance lambda: `2.0`, reduction: `sum`, observed fraction: `0.15`.

## Reconstruction summary

| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |
|---:|---:|---:|---:|:---:|:---:|
| 10 | 0.2364 | 0.0928 | 0.0946 | True | True |
| 20 | 0.2276 | 0.0936 | 0.0952 | True | True |
| 50 | 0.2241 | 0.0946 | 0.0956 | True | True |
| 100 | 0.2231 | 0.0949 | 0.0958 | True | True |

## Mechanism diagnostics

- Train-coupled displacement, Gauss: `49.5727`
- Train-coupled displacement, Info: `2.1694`
- Train-coupled curvature proxy, Gauss: `0.0311`
- Train-coupled curvature proxy, Info: `0.0195`
- Marginal inference source displacement to current eval target: `47.8945`
- Warm inference source displacement to current eval target: `2.7309`
- Learned inference source displacement to current eval target: `3.1735`
- Learned inference source nRMSE to current eval target: `0.1119`

The marginal source is distribution-matched but not target-short. The warm source is a fixed measurement-informed low-frequency reconstruction from `y`, not an oracle `S(X_true)`.

## Gate

Pass all matched NFE: `True`
