# S3GM-PF-ODE attribution baseline

Eval split: `test`, windows: `8`, seeds: `0`.
Mask seed: `0`, observed fraction: `0.15`.
Guidance lambda: `2.0`, reduction: `sum`.

| NFE | S3GM-PF guided | S3GM-PF unguided | S3FM learned guided | S3FM learned unguided | S3FM<=PF |
|---:|---:|---:|---:|---:|:---:|
| 10 | 1947715.2500 | 1.2270 | 0.0509 | 0.0558 | True |
| 20 | 6327378944.0000 | 1.2564 | 0.0509 | 0.0557 | True |

## Gate

Pass S3FM learned guided vs S3GM-PF guided at all NFE: `True`
