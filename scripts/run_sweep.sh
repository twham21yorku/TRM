#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/deepglobe_default.yaml}
SEEDS=(1337 2021 42 7 99)

for SEED in "${SEEDS[@]}"; do
  OUT="experiments/run_seed_${SEED}"
  echo "=== Running seed ${SEED} → ${OUT} ==="
  python -m src.data.patchify --cfg "$CFG" --root $(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['dataset']['root'])")
  python -m src.train --cfg "$CFG" --seed ${SEED} --out_dir "$OUT"
done

echo "Sweep complete. Logs in experiments/run_seed_*/train_log.csv"
