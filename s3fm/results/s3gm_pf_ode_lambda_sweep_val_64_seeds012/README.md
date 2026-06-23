# S3GM-PF-ODE lambda sweep baseline

Eval split: `val`, windows: `64`, seeds: `0,1,2`.
Mask seed: `0`, observed fraction: `0.15`.
PF lambdas: `0,0.0001,0.001,0.01,0.03,0.1,0.25,0.5,1.0`.
S3FM lambda: `2.0`, reduction: `sum`.

The best PF lambda is selected per NFE on this run. This is optimistic for the PF baseline.

| NFE | best PF lambda | best PF guided | PF unguided | S3FM learned guided | S3FM unguided | S3FM<=best PF |
|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 0.1 | 1.0394 | 1.2084 | 0.0976 | 0.0978 | True |
| 20 | 0.25 | 0.8674 | 1.2295 | 0.0956 | 0.0988 | True |
| 50 | 0.5 | 0.6544 | 1.2476 | 0.0964 | 0.0996 | True |
| 100 | 1 | 0.4375 | 1.2548 | 0.0967 | 0.0999 | True |

## Gate

Pass S3FM learned guided vs best PF at all NFE: `True`
