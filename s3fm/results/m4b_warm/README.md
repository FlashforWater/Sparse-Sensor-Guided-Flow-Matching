# M4b three-way ablation

Info inference source: `warm`.
Eval split: `test`, windows: `8`, seeds: `0`.
Guidance lambda: `5.0`, reduction: `sum`, observed fraction: `0.15`.

## Reconstruction summary

| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |
|---:|---:|---:|---:|:---:|:---:|
| 10 | 0.0968 | 0.0157 | 0.0212 | True | True |
| 20 | 0.0958 | 0.0157 | 0.0211 | True | True |
| 50 | 0.0955 | 0.0157 | 0.0211 | True | True |
| 100 | 0.0953 | 0.0157 | 0.0211 | True | True |

## Mechanism diagnostics

- Train-coupled displacement, Gauss: `47.5190`
- Train-coupled displacement, Info: `1.8332`
- Train-coupled curvature proxy, Gauss: `0.0304`
- Train-coupled curvature proxy, Info: `0.0193`
- Marginal inference source displacement to current eval target: `42.5471`
- Warm inference source displacement to current eval target: `1.9362`

The marginal source is distribution-matched but not target-short. The warm source is a fixed measurement-informed low-frequency reconstruction from `y`, not an oracle `S(X_true)`.

## Gate

Pass all matched NFE: `True`
