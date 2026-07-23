#!/usr/bin/env bash
# Launch the MimicGen 12-task multitask token-grouping ablations.
#
# Each entry is one full training run (own savepath prefix, no collisions).
# Run from the inner `diffuser/` dir so `config` + `diffuser` are importable,
# and add the repo root (..) for `dataset_paths`. One GPU per run.
#
# Usage:
#   ./scripts/run_token_ablations.sh                 # list the runs
#   ./scripts/run_token_ablations.sh randgroupA      # launch one run
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_token_ablations.sh noproprio &
#
# The uniform-per-dim baseline (config.mimicgen_multitask_dlp) is already done
# and intentionally not listed here.
set -euo pipefail

# cd into this script's parent's parent = EC-Diffuser/diffuser
cd "$(dirname "$0")/.."
export PYTHONPATH=".:..:${PYTHONPATH:-}"

declare -A CONFIGS=(
  [randgroupA]="config.mimicgen_multitask_randgroupA_dlp"            # action [2,1,3,1] / proprio [4,1,2,3]
  [randgroupB]="config.mimicgen_multitask_randgroupB_dlp"            # action [3,1,1,2] / proprio [2,3,1,4]
  [noproprio]="config.mimicgen_multitask_noproprio_dlp"             # per-dim action, NO proprio
  [tokenaction_singleprop]="config.mimicgen_multitask_tokenaction_singleprop_dlp"  # [3,3,1] action, 1 proprio token
  # NOTE: singleaction_tokenprop (1 action token + [3,6,1] proprio) is intentionally
  # omitted -- it is token-for-token identical to the already-trained `singleaction`
  # run (config.mimicgen_multitask_singleaction_dlp). Cite that run instead of re-training.
)

if [[ $# -eq 0 ]]; then
  echo "Available runs (pass one as an argument):"
  for k in "${!CONFIGS[@]}"; do printf '  %-24s %s\n' "$k" "${CONFIGS[$k]}"; done
  exit 0
fi

name="$1"
cfg="${CONFIGS[$name]:-}"
if [[ -z "$cfg" ]]; then
  echo "Unknown run '$name'. Options: ${!CONFIGS[*]}" >&2
  exit 1
fi

echo "[run_token_ablations] launching $name -> $cfg on GPU ${CUDA_VISIBLE_DEVICES:-0}"
exec python scripts/train.py --config "$cfg" --num_entity 12 --input_type dlp
