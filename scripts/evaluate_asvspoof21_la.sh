#!/usr/bin/env bash
set -euo pipefail

: "${XLSR_CHECKPOINT:?Set XLSR_CHECKPOINT to xlsr2_300m.pt}"
: "${ASVSPOOF_ROOT:?Set ASVSPOOF_ROOT to the dataset directory}"
: "${PROTOCOL_ROOT:?Set PROTOCOL_ROOT to the protocol directory}"

python generate_scores.py \
  --xlsr-checkpoint "$XLSR_CHECKPOINT" \
  --checkpoint checkpoints/cmsd/best.pt \
  --data-root "$ASVSPOOF_ROOT/ASVspoof2021_LA_eval/flac" \
  --protocol "$PROTOCOL_ROOT/ASVspoof2021.LA.cm.eval.trl.txt" \
  --protocol-format asvspoof \
  --audio-extension .flac \
  --output scores/asvspoof2021_la.txt
