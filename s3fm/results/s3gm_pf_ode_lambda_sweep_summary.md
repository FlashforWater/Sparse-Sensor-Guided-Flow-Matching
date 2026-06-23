# S3GM-PF-ODE lambda sweep summary

This is the first optimistic attribution baseline for S3GM-PF-ODE.

## Setup

- Score prior: `experiments/kse_score_prior/best.pt`
- S3FM learned-source prior: `experiments/kse_prior_info/final.pt`
- Source inference: `experiments/kse_source_inference/best.pt`
- Eval split: test
- Eval windows: 64
- Sparse observation fraction: 0.15
- Mask seed: 0
- Source seeds: 0,1,2
- S3FM guidance lambda: 2.0
- PF lambdas swept: 0, 0.1, 0.25, 0.5, 1.0

The PF baseline is given an optimistic advantage: for each NFE, the best PF
lambda is selected from the sweep on this run.

## Result

S3FM learned-source guided reconstruction beats the best PF-ODE guided baseline
at every matched NFE.

| NFE | best PF lambda | best PF guided nRMSE | PF unguided nRMSE | S3FM learned guided nRMSE | S3FM learned unguided nRMSE |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.1 | 1.0384 +- 0.0113 | 1.2066 | 0.0900 | 0.0942 |
| 20 | 0.25 | 0.8657 +- 0.0088 | 1.2286 | 0.0907 | 0.0947 |
| 50 | 0.5 | 0.6432 +- 0.0073 | 1.2474 | 0.0914 | 0.0952 |
| 100 | 1.0 | 0.4173 +- 0.0037 | 1.2549 | 0.0916 | 0.0953 |

Gate:

- S3FM learned guided <= best S3GM-PF guided at all NFE: True
- S3FM learned guided <= S3FM learned unguided at all NFE: True
- Best S3GM-PF guided <= S3GM-PF unguided at all NFE: True

## Interpretation

This weakens the objection that S3FM's improvement is merely "ODE instead of
SDE". Even when PF-ODE gets its own per-NFE lambda sweep, it remains much worse
than the learned observation-informed source flow.

This is still not the final paper table. Two caveats remain:

1. The PF lambda was selected on the evaluation run, which is optimistic for PF.
   Final reporting should tune PF lambda on validation and report test.
2. The score prior is a local VP epsilon baseline, not the original S3GM author's
   released model. A paper-grade comparison should either reproduce S3GM more
   closely or clearly label this as an internal PF-ODE attribution baseline.
