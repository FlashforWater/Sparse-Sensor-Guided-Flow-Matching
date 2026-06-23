# S3GM-PF-ODE lambda sweep baseline

Eval split: `test`, windows: `8`, seeds: `0`.
Mask seed: `0`, observed fraction: `0.15`.
PF lambdas: `0,0.0001,0.001,0.01,0.1,0.25,0.5,1.0`.
PF lambda by NFE: `10:0.25`.
S3FM lambda: `2.0`, reduction: `sum`.

The PF lambda is fixed per NFE from `--pf-lambda-by-nfe`.

| NFE | best PF lambda | best PF guided | PF unguided | S3FM learned guided | S3FM unguided | S3FM<=best PF |
|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 0.25 | 0.8746 | 1.2270 | 0.0509 | 0.0558 | True |

## Gate

Pass S3FM learned guided vs best PF at all NFE: `True`
