# S3GM-PF-ODE lambda sweep baseline

Eval split: `test`, windows: `8`, seeds: `0`.
Mask seed: `0`, observed fraction: `0.15`.
PF lambdas: `0,0.01,0.1,0.5`.
S3FM lambda: `2.0`, reduction: `sum`.

The best PF lambda is selected per NFE on this run. This is optimistic for the PF baseline.

| NFE | best PF lambda | best PF guided | PF unguided | S3FM learned guided | S3FM unguided | S3FM<=best PF |
|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 0.1 | 1.0585 | 1.2270 | 0.0509 | 0.0558 | True |
| 20 | 0.5 | 0.6684 | 1.2564 | 0.0509 | 0.0557 | True |

## Gate

Pass S3FM learned guided vs best PF at all NFE: `True`
