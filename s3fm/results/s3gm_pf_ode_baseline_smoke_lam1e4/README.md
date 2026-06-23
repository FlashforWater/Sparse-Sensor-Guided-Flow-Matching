# S3GM-PF-ODE attribution baseline

Eval split: `test`, windows: `8`, seeds: `0`.
Mask seed: `0`, observed fraction: `0.15`.
Guidance lambda: `0.0001`, reduction: `sum`.

| NFE | S3GM-PF guided | S3GM-PF unguided | S3FM learned guided | S3FM learned unguided | S3FM<=PF |
|---:|---:|---:|---:|---:|:---:|
| 10 | 1.2267 | 1.2270 | 0.0558 | 0.0558 | True |
| 20 | 1.2562 | 1.2564 | 0.0557 | 0.0557 | True |

## Gate

Pass S3FM learned guided vs S3GM-PF guided at all NFE: `True`
