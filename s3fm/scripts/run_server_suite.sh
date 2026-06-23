#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root:
#   bash scripts/run_server_suite.sh
#
# Common overrides:
#   DEVICE=cuda WINDOWS=256 MASK_SEEDS=0,1,2,3,4 bash scripts/run_server_suite.sh
#   OBS_FRACTIONS=0.05,0.1,0.15,0.3 bash scripts/run_server_suite.sh

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-results/server_suite}"
OBS_FRACTIONS="${OBS_FRACTIONS:-0.15}"
MASK_SEEDS="${MASK_SEEDS:-0,1,2}"
SOURCE_SEEDS="${SOURCE_SEEDS:-0,1,2}"
WINDOWS="${WINDOWS:-128}"
NFE="${NFE:-10,20,50,100}"
S3FM_LAMBDA="${S3FM_LAMBDA:-2.0}"
PF_LAMBDAS="${PF_LAMBDAS:-0,0.0001,0.001,0.01,0.03,0.1,0.25,0.5,1.0}"

GAUSS_CKPT="${GAUSS_CKPT:-experiments/kse_prior/final.pt}"
INFO_CKPT="${INFO_CKPT:-experiments/kse_prior_info/final.pt}"
SCORE_CKPT="${SCORE_CKPT:-experiments/kse_score_prior/best.pt}"
SOURCE_INFERENCE_CKPT="${SOURCE_INFERENCE_CKPT:-experiments/kse_source_inference/best.pt}"

EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-false}" == "true" ]]; then
  EXTRA_ARGS+=("--dry-run")
fi
if [[ "${OVERWRITE:-0}" == "1" || "${OVERWRITE:-false}" == "true" ]]; then
  EXTRA_ARGS+=("--overwrite")
fi
if [[ "${SKIP_M4B:-0}" == "1" || "${SKIP_M4B:-false}" == "true" ]]; then
  EXTRA_ARGS+=("--skip-m4b")
fi
if [[ "${SKIP_PF:-0}" == "1" || "${SKIP_PF:-false}" == "true" ]]; then
  EXTRA_ARGS+=("--skip-pf")
fi

"${PYTHON_BIN}" -m s3fm.run_server_suite \
  --out "${OUT}" \
  --device "${DEVICE}" \
  --observed-fractions "${OBS_FRACTIONS}" \
  --mask-seeds "${MASK_SEEDS}" \
  --seeds "${SOURCE_SEEDS}" \
  --num-windows "${WINDOWS}" \
  --nfe "${NFE}" \
  --s3fm-lambda "${S3FM_LAMBDA}" \
  --pf-lambdas "${PF_LAMBDAS}" \
  --gauss-ckpt "${GAUSS_CKPT}" \
  --info-ckpt "${INFO_CKPT}" \
  --score-ckpt "${SCORE_CKPT}" \
  --source-inference-ckpt "${SOURCE_INFERENCE_CKPT}" \
  "${EXTRA_ARGS[@]}"
