# S3GM-PF-ODE baseline smoke summary

This records the first implementation check for the S3GM-PF-ODE attribution
baseline.

## What was implemented

- VP epsilon-prediction score prior training.
- Reverse probability-flow ODE velocity.
- PF-ODE measurement guidance through the denoised endpoint estimate `x0_hat`.
- A baseline runner comparing:
  - `A_s3gm_pf_guided`
  - `A0_s3gm_pf_unguided`
  - `B_s3fm_learned_guided`
  - `C_s3fm_learned_unguided`

Score prior checkpoint:

- `experiments/kse_score_prior/best.pt`
- 4000 training steps on MPS
- final/best validation denoise nRMSE: `0.4386` at `t=0.5`
- final validation epsilon loss: `0.0179`

## Guidance-scale finding

PF-ODE guidance has a very different scale from flow-matching guidance.

- `lambda0=2.0` explodes for PF-ODE.
- `lambda0=1e-4` to `1e-2` is stable but too weak.
- `lambda0=0.1` improves PF-ODE but remains far behind S3FM.
- `lambda0=0.5` or `1.0` can help at larger NFE but explodes at low NFE.

This means PF-ODE needs its own lambda sweep. Reusing the S3FM lambda is not a
fair baseline.

## Small 8-window smoke results

Best PF numbers observed in the quick sweep remain far worse than S3FM learned
source:

| NFE | best observed PF guided nRMSE | S3FM learned guided nRMSE in same runs |
|---:|---:|---:|
| 10 | ~1.06 (`lambda0=0.1`) | ~0.055 |
| 20 | ~0.67 (`lambda0=0.5`) | ~0.054 |
| 50 | ~0.43 (`lambda0=1.0`) | ~0.053 |

These are smoke results only. They show that the baseline code runs, but they
are not yet the final paper-level S3GM-PF-ODE comparison.

## Next required step

Make PF-ODE baseline fairer before drawing a paper claim:

1. Add a PF-only lambda sweep runner that can select the best PF lambda per NFE
   without changing the S3FM lambda.
2. Run the sweep on 64 windows and several mask seeds.
3. If PF remains weak, improve the score baseline or sampler before reporting it
   as a final attribution baseline.
