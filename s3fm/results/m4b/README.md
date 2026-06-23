# M4b three-way ablation

Info inference source: `marginal`.
Eval split: `test`, windows: `8`, seeds: `0`.
Guidance lambda: `5.0`, reduction: `sum`, observed fraction: `0.15`.

## Reconstruction summary

| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |
|---:|---:|---:|---:|:---:|:---:|
| 10 | 0.0968 | 1.4283 | 1.3988 | False | False |
| 20 | 0.0958 | 1.3138 | 1.3988 | False | True |
| 50 | 0.0955 | 1.2677 | 1.3988 | False | True |
| 100 | 0.0953 | 1.2838 | 1.3988 | False | True |

## Mechanism diagnostics

- Train-coupled displacement, Gauss: `47.5190`
- Train-coupled displacement, Info: `1.8332`
- Train-coupled curvature proxy, Gauss: `0.0304`
- Train-coupled curvature proxy, Info: `0.0193`
- Marginal inference source displacement to current eval target: `42.5471`

The marginal-source displacement is not expected to be target-short because it is not `S(X_true)`; it is logged to make the inference-source distinction explicit.

## Gate

Pass all matched NFE: `False`
