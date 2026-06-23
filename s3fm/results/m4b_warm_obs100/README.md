# M4b three-way ablation

Info inference source: `warm`.
Eval split: `test`, windows: `8`, seeds: `0`.
Guidance lambda: `5.0`, reduction: `sum`, observed fraction: `1.0`.

## Reconstruction summary

| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |
|---:|---:|---:|---:|:---:|:---:|
| 20 | 0.0169 | 0.0027 | 0.0144 | True | True |
| 50 | 0.0131 | 0.0027 | 0.0144 | True | True |

## Mechanism diagnostics

- Train-coupled displacement, Gauss: `47.5190`
- Train-coupled displacement, Info: `1.8332`
- Train-coupled curvature proxy, Gauss: `0.0304`
- Train-coupled curvature proxy, Info: `0.0193`
- Marginal inference source displacement to current eval target: `42.5471`
- Warm inference source displacement to current eval target: `1.8332`

The marginal source is distribution-matched but not target-short. The warm source is a fixed measurement-informed reconstruction from `y`, not an oracle `S(X_true)`.

## Gate

Pass all matched NFE: `True`
