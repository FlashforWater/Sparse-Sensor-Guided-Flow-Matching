# M4b three-way ablation

Info inference source: `warm`.
Eval split: `test`, windows: `64`, seeds: `0`.
Guidance lambda: `5.0`, reduction: `sum`, observed fraction: `0.15`.

## Reconstruction summary

| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |
|---:|---:|---:|---:|:---:|:---:|
| 10 | 0.1151 | 0.2912 | 0.0968 | False | False |
| 20 | 0.1193 | 0.1923 | 0.0979 | False | False |
| 50 | 0.1169 | 0.0965 | 0.0986 | True | True |
| 100 | 0.1173 | 0.0947 | 0.0989 | True | True |

## Mechanism diagnostics

- Train-coupled displacement, Gauss: `49.5046`
- Train-coupled displacement, Info: `2.1703`
- Train-coupled curvature proxy, Gauss: `0.0311`
- Train-coupled curvature proxy, Info: `0.0195`
- Marginal inference source displacement to current eval target: `46.5664`
- Warm inference source displacement to current eval target: `2.6268`

The marginal source is distribution-matched but not target-short. The warm source is a fixed measurement-informed low-frequency reconstruction from `y`, not an oracle `S(X_true)`.

## Gate

Pass all matched NFE: `False`
