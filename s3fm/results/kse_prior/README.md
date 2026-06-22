# KSE flow prior — training record

Trained locally (Apple MPS), ~21 min, 4000 steps. Config:
`configs/kse_flow_prior.yaml`. Model: VideoUNetVelocity1D, 0.34M params.

Result: val FM loss 2.43 -> 0.22, endpoint nRMSE (s=0.9) 0.148 -> 0.029.

Checkpoints (`best.pt`, `final.pt`, `step_*.pt`) are NOT committed (multi-MB,
gitignored). Reproduce with:

    python -m s3fm.train_flow --config configs/kse_flow_prior.yaml --out experiments/kse_prior

Each checkpoint stores: model + EMA weights, normalization mean/std, resolved
config, step, metric.
