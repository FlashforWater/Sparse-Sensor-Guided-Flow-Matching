# S3GM-PF-ODE attribution baseline

Eval split: `test`, windows: `8`, seeds: `0`.
Mask seed: `0`, observed fraction: `0.15`.
Guidance lambda: `1.0`, reduction: `sum`.

| NFE | S3GM-PF guided | S3GM-PF unguided | S3FM learned guided | S3FM learned unguided | S3FM<=PF |
|---:|---:|---:|---:|---:|:---:|
| 10 | 20710.0801 | 1.2270 | 0.0530 | 0.0558 | True |
| 20 | 23098748.0000 | 1.2564 | 0.0529 | 0.0557 | True |
| 50 | 0.4346 | 1.2810 | 0.0529 | 0.0556 | True |

## Gate

Pass S3FM learned guided vs S3GM-PF guided at all NFE: `True`
