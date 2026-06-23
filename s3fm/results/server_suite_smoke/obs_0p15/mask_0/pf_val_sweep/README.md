# S3GM-PF-ODE lambda sweep baseline

Eval split: `val`, windows: `8`, seeds: `0`.
Mask seed: `0`, observed fraction: `0.15`.
PF lambdas: `0,0.1,0.25`.
PF lambda by NFE: `None`.
S3FM lambda: `2.0`, reduction: `sum`.

The best PF lambda is selected per NFE on this run. This is optimistic for the PF baseline.

| NFE | best PF lambda | best PF guided | PF unguided | S3FM learned guided | S3FM unguided | S3FM<=best PF |
|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 0.25 | 0.9115 | 1.2531 | 0.0483 | 0.0529 | True |

## Gate

Pass S3FM learned guided vs best PF at all NFE: `True`
