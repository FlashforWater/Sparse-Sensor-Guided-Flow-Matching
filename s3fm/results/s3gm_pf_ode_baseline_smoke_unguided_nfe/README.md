# S3GM-PF-ODE attribution baseline

Eval split: `test`, windows: `8`, seeds: `0`.
Mask seed: `0`, observed fraction: `0.15`.
Guidance lambda: `0.0`, reduction: `sum`.

| NFE | S3GM-PF guided | S3GM-PF unguided | S3FM learned guided | S3FM learned unguided | S3FM<=PF |
|---:|---:|---:|---:|---:|:---:|
| 10 | 1.2270 | 1.2270 | 0.0558 | 0.0558 | True |
| 20 | 1.2564 | 1.2564 | 0.0557 | 0.0557 | True |
| 50 | 1.2810 | 1.2810 | 0.0556 | 0.0556 | True |
| 100 | 1.2906 | 1.2906 | 0.0556 | 0.0556 | True |
| 500 | 1.2987 | 1.2987 | 0.0556 | 0.0556 | True |

## Gate

Pass S3FM learned guided vs S3GM-PF guided at all NFE: `True`
