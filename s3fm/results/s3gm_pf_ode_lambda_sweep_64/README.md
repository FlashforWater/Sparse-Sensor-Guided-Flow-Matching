# S3GM-PF-ODE lambda sweep baseline

Eval split: `test`, windows: `64`, seeds: `0`.
Mask seed: `0`, observed fraction: `0.15`.
PF lambdas: `0,0.0001,0.001,0.01,0.03,0.1,0.25,0.5,1.0`.
S3FM lambda: `2.0`, reduction: `sum`.

The best PF lambda is selected per NFE on this run. This is optimistic for the PF baseline.

| NFE | best PF lambda | best PF guided | PF unguided | S3FM learned guided | S3FM unguided | S3FM<=best PF |
|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 0.25 | 0.8386 | 1.1952 | 0.0896 | 0.0942 | True |
| 20 | 0.5 | 0.6265 | 1.2168 | 0.0905 | 0.0947 | True |
| 50 | 1 | 0.4110 | 1.2352 | 0.0912 | 0.0951 | True |
| 100 | 1 | 0.4121 | 1.2426 | 0.0915 | 0.0953 | True |

## Gate

Pass S3FM learned guided vs best PF at all NFE: `True`
