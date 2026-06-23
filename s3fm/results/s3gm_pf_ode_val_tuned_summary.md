# S3GM-PF-ODE validation-tuned test summary

This is the cleaner PF-ODE attribution baseline:

1. Tune PF guidance lambda on validation.
2. Fix the selected lambda per NFE.
3. Report on test without re-selecting lambda.

## Validation tuning

Validation sweep:

- Directory: `results/s3gm_pf_ode_lambda_sweep_val_64_seeds012`
- Eval split: val
- Windows: 64
- Seeds: 0,1,2
- Mask seed: 0
- PF lambdas swept: 0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.25, 0.5, 1.0

Selected PF lambdas:

| NFE | selected PF lambda |
|---:|---:|
| 10 | 0.1 |
| 20 | 0.25 |
| 50 | 0.5 |
| 100 | 1.0 |

## Test report

Test run:

- Directory: `results/s3gm_pf_ode_val_tuned_test_64_seeds012`
- Eval split: test
- Windows: 64
- Seeds: 0,1,2
- Mask seed: 0
- PF lambdas fixed from validation
- S3FM learned-source lambda: 2.0

| NFE | val-selected PF lambda | S3GM-PF guided nRMSE | S3GM-PF unguided nRMSE | S3FM learned guided nRMSE | S3FM learned unguided nRMSE |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.1 | 1.0384 +- 0.0113 | 1.2066 | 0.0900 | 0.0942 |
| 20 | 0.25 | 0.8657 +- 0.0088 | 1.2286 | 0.0907 | 0.0947 |
| 50 | 0.5 | 0.6432 +- 0.0073 | 1.2474 | 0.0914 | 0.0952 |
| 100 | 1.0 | 0.4173 +- 0.0037 | 1.2549 | 0.0916 | 0.0953 |

Gate:

- S3FM learned guided <= val-tuned S3GM-PF guided at all NFE: True
- S3FM learned guided <= S3FM learned unguided at all NFE: True
- Val-tuned S3GM-PF guided <= S3GM-PF unguided at all NFE: True

## Interpretation

This addresses the immediate attribution concern that S3FM only wins because it
uses an ODE sampler. The PF-ODE score baseline also uses a deterministic ODE and
gets its guidance strength tuned on validation, yet remains much worse than the
learned observation-informed source flow on test.

Remaining caveat:

- This is still an internal VP score-prior baseline, not the original S3GM
  authors' exact released model. It is suitable as an attribution control, but a
  final paper should label it clearly or add a closer S3GM reproduction.
