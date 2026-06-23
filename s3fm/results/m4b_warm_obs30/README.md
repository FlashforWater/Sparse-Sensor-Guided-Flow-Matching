# M4b three-way ablation

Info inference source: `warm`.
Eval split: `test`, windows: `8`, seeds: `0`.
Guidance lambda: `5.0`, reduction: `sum`, observed fraction: `0.3`.

## Reconstruction summary

| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |
|---:|---:|---:|---:|:---:|:---:|
| 20 | 0.0339 | 0.1319 | 0.1389 | False | True |
| 50 | 0.0355 | 0.1328 | 0.1398 | False | True |

## Mechanism diagnostics

- Train-coupled displacement, Gauss: `47.5190`
- Train-coupled displacement, Info: `1.8332`
- Train-coupled curvature proxy, Gauss: `0.0304`
- Train-coupled curvature proxy, Info: `0.0193`
- Marginal inference source displacement to current eval target: `42.5471`
- Warm inference source displacement to current eval target: `5.0108`

The marginal source is distribution-matched but not target-short. The warm source is a fixed measurement-informed reconstruction from `y`, not an oracle `S(X_true)`.

## Gate

Pass all matched NFE: `False`
