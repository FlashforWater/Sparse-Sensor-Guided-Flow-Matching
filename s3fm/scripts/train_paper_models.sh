#!/usr/bin/env bash
set -euo pipefail

# Run from the s3fm project root:
#   bash scripts/train_paper_models.sh
#
# Optional controls:
#   DEVICE=cuda EXPERIMENT_ROOT=experiments/paper bash scripts/train_paper_models.sh
#   DRY_RUN=1 bash scripts/train_paper_models.sh
#   RUN_SCORE=0 bash scripts/train_paper_models.sh
#   GAUSS_STEPS=50000 INFO_STEPS=50000 SCORE_STEPS=50000 SOURCE_STEPS=40000 bash scripts/train_paper_models.sh

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
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-experiments/paper}"

RUN_GAUSS="${RUN_GAUSS:-1}"
RUN_INFO="${RUN_INFO:-1}"
RUN_SCORE="${RUN_SCORE:-1}"
RUN_SOURCE="${RUN_SOURCE:-1}"

GAUSS_STEPS="${GAUSS_STEPS:-}"
INFO_STEPS="${INFO_STEPS:-}"
SCORE_STEPS="${SCORE_STEPS:-}"
SOURCE_STEPS="${SOURCE_STEPS:-}"

run_train() {
  local module="$1"
  local config="$2"
  local out="$3"
  local steps="$4"
  local args=("--config" "$config" "--out" "$out" "--device" "$DEVICE")
  if [[ -n "$steps" ]]; then
    args+=("--steps" "$steps")
  fi
  echo "[train] ${module} -> ${out}"
  if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-false}" == "true" ]]; then
    printf '[dry-run]'
    printf ' %q' "${PYTHON_BIN}" -m "$module" "${args[@]}"
    printf '\n'
    return
  fi
  "${PYTHON_BIN}" -m "$module" "${args[@]}"
}

mkdir -p "${EXPERIMENT_ROOT}"

if [[ "$RUN_GAUSS" == "1" ]]; then
  run_train \
    "s3fm.train_flow" \
    "configs/paper_kse_flow_prior.yaml" \
    "${EXPERIMENT_ROOT}/kse_prior" \
    "$GAUSS_STEPS"
fi

if [[ "$RUN_INFO" == "1" ]]; then
  run_train \
    "s3fm.train_flow" \
    "configs/paper_kse_flow_prior_info.yaml" \
    "${EXPERIMENT_ROOT}/kse_prior_info" \
    "$INFO_STEPS"
fi

if [[ "$RUN_SCORE" == "1" ]]; then
  run_train \
    "s3fm.train_score" \
    "configs/paper_kse_score_prior.yaml" \
    "${EXPERIMENT_ROOT}/kse_score_prior" \
    "$SCORE_STEPS"
fi

if [[ "$RUN_SOURCE" == "1" ]]; then
  run_train \
    "s3fm.train_source_inference" \
    "configs/paper_kse_source_inference.yaml" \
    "${EXPERIMENT_ROOT}/kse_source_inference" \
    "$SOURCE_STEPS"
fi

echo "paper checkpoints expected at:"
echo "  ${EXPERIMENT_ROOT}/kse_prior/final.pt"
echo "  ${EXPERIMENT_ROOT}/kse_prior_info/final.pt"
echo "  ${EXPERIMENT_ROOT}/kse_score_prior/best.pt"
echo "  ${EXPERIMENT_ROOT}/kse_source_inference/best.pt"
