# M4b three-way ablation

Info inference source: `learned`.
Eval split: `test`, windows: `1`, seeds: `0`.
Guidance lambda: `5.0`, reduction: `sum`, observed fraction: `0.15`.

## Reconstruction summary

| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |
|---:|---:|---:|---:|:---:|:---:|
| 2 | 0.1868 | 1.4268 | 0.8734 | False | False |

## Mechanism diagnostics

- Train-coupled displacement, Gauss: `46.7897`
- Train-coupled displacement, Info: `1.8961`
- Train-coupled curvature proxy, Gauss: `0.1504`
- Train-coupled curvature proxy, Info: `0.1243`
- Marginal inference source displacement to current eval target: `46.7697`
- Warm inference source displacement to current eval target: `1.9972`

The marginal source is distribution-matched but not target-short. The warm source is a fixed measurement-informed low-frequency reconstruction from `y`, not an oracle `S(X_true)`.

## Gate

Pass all matched NFE: `False`
