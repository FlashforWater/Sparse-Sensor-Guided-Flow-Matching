# M4b three-way ablation

Info inference source: `marginal`.
Eval split: `test`, windows: `1`, seeds: `0`.
Guidance lambda: `5.0`, reduction: `sum`, observed fraction: `0.15`.

## Reconstruction summary

| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |
|---:|---:|---:|---:|:---:|:---:|
| 2 | 0.1868 | 2.1172 | 1.6083 | False | False |

## Mechanism diagnostics

- Train-coupled displacement, Gauss: `46.7897`
- Train-coupled displacement, Info: `1.8961`
- Train-coupled curvature proxy, Gauss: `0.1504`
- Train-coupled curvature proxy, Info: `0.1243`
- Marginal inference source displacement to current eval target: `46.7697`

The marginal-source displacement is not expected to be target-short because it is not `S(X_true)`; it is logged to make the inference-source distinction explicit.

## Gate

Pass all matched NFE: `False`
