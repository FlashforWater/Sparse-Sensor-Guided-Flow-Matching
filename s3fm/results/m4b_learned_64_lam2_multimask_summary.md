# M4b learned source multi-mask summary

This is a compact summary of the learned observation-informed source experiment.

Setup:

- Eval split: test
- Eval windows: 64
- Source seeds: 0,1,2
- Mask seeds: 0,1,2
- Observed fraction: 0.15
- Info source: learned `q_phi(y,H)`
- Source inference checkpoint: `experiments/kse_source_inference/best.pt`
- Guidance: `lambda0=2.0`, `reduction=sum`
- NFE: 10,20,50,100

## Gate result

All three mask seeds pass the M4b gate:

| mask seed | pass all NFE | result directory |
|---:|:---:|---|
| 0 | True | `results/m4b_learned_64_lam2_seeds012` |
| 1 | True | `results/m4b_learned_64_lam2_seeds012_mask1` |
| 2 | True | `results/m4b_learned_64_lam2_seeds012_mask2` |

The gate requires both:

- learned Info guided <= Gaussian guided at matched NFE
- learned Info guided <= learned Info unguided at matched NFE

## Reconstruction nRMSE

### mask seed 0

| NFE | Gaussian guided | learned Info guided | learned Info unguided |
|---:|---:|---:|---:|
| 10 | 0.2521 | 0.0900 | 0.0942 |
| 20 | 0.2420 | 0.0907 | 0.0947 |
| 50 | 0.2374 | 0.0914 | 0.0952 |
| 100 | 0.2361 | 0.0916 | 0.0953 |

### mask seed 1

| NFE | Gaussian guided | learned Info guided | learned Info unguided |
|---:|---:|---:|---:|
| 10 | 0.2816 | 0.0967 | 0.1003 |
| 20 | 0.2707 | 0.0976 | 0.1008 |
| 50 | 0.2660 | 0.0983 | 0.1012 |
| 100 | 0.2646 | 0.0986 | 0.1013 |

### mask seed 2

| NFE | Gaussian guided | learned Info guided | learned Info unguided |
|---:|---:|---:|---:|
| 10 | 0.2364 | 0.0928 | 0.0946 |
| 20 | 0.2276 | 0.0936 | 0.0952 |
| 50 | 0.2241 | 0.0946 | 0.0956 |
| 100 | 0.2231 | 0.0949 | 0.0958 |

## Mechanism diagnostics

The learned source is much closer to the current target than a marginal source,
while still being restricted to the low-bandwidth source space rather than a full
field reconstruction.

| mask seed | marginal source displacement | warm source displacement | learned source displacement | learned source nRMSE |
|---:|---:|---:|---:|---:|
| 0 | 47.8945 | 2.6269 | 3.2565 | 0.1121 |
| 1 | 47.8945 | 2.7434 | 3.4588 | 0.1175 |
| 2 | 47.8945 | 2.7309 | 3.1735 | 0.1119 |

## Interpretation

The current evidence supports the learned-source version of the idea:

- `q_phi(y,H)` can infer a useful observation-conditioned low-bandwidth source.
- The learned source is target-short compared with Gaussian or marginal sources.
- Measurement guidance helps the learned-source prior at matched NFE.
- The result is stable across three source seeds and three sparse-mask seeds.

This is still not the final paper-level gate. The next required attribution
baseline is `S3GM-PF-ODE`, which is not present in the repository yet.
