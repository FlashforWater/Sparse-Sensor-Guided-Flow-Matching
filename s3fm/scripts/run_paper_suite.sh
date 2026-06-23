#!/usr/bin/env bash
set -euo pipefail

# Run from the s3fm project root after training paper checkpoints:
#   bash scripts/run_paper_suite.sh

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-experiments/paper}"

export DEVICE="${DEVICE:-cuda}"
export OUT="${OUT:-results/paper_server_suite}"
export OBS_FRACTIONS="${OBS_FRACTIONS:-0.05,0.1,0.15,0.3}"
export MASK_SEEDS="${MASK_SEEDS:-0,1,2}"
export SOURCE_SEEDS="${SOURCE_SEEDS:-0,1,2}"
export WINDOWS="${WINDOWS:-256}"
export NFE="${NFE:-10,20,50,100}"
export S3FM_LAMBDA="${S3FM_LAMBDA:-2.0}"
export PF_LAMBDAS="${PF_LAMBDAS:-0,0.0001,0.001,0.01,0.03,0.1,0.25,0.5,1.0}"

export GAUSS_CKPT="${GAUSS_CKPT:-${EXPERIMENT_ROOT}/kse_prior/final.pt}"
export INFO_CKPT="${INFO_CKPT:-${EXPERIMENT_ROOT}/kse_prior_info/final.pt}"
export SCORE_CKPT="${SCORE_CKPT:-${EXPERIMENT_ROOT}/kse_score_prior/best.pt}"
export SOURCE_INFERENCE_CKPT="${SOURCE_INFERENCE_CKPT:-${EXPERIMENT_ROOT}/kse_source_inference/best.pt}"

bash scripts/run_server_suite.sh
