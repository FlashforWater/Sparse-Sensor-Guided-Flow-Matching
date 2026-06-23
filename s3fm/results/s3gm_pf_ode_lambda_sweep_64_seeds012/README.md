# S3GM-PF-ODE lambda sweep baseline

Eval split: `test`, windows: `64`, seeds: `0,1,2`.
Mask seed: `0`, observed fraction: `0.15`.
PF lambdas: `0,0.1,0.25,0.5,1.0`.
S3FM lambda: `2.0`, reduction: `sum`.

The best PF lambda is selected per NFE on this run. This is optimistic for the PF baseline.

| NFE | best PF lambda | best PF guided | PF unguided | S3FM learned guided | S3FM unguided | S3FM<=best PF |
|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 0.1 | 1.0384 | 1.2066 | 0.0900 | 0.0942 | True |
| 20 | 0.25 | 0.8657 | 1.2286 | 0.0907 | 0.0947 | True |
| 50 | 0.5 | 0.6432 | 1.2474 | 0.0914 | 0.0952 | True |
| 100 | 1 | 0.4173 | 1.2549 | 0.0916 | 0.0953 | True |

## Gate

Pass S3FM learned guided vs best PF at all NFE: `True`
