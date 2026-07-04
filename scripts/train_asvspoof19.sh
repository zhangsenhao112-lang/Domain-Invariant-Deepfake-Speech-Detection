#!/usr/bin/env bash
set -euo pipefail

: "${XLSR_CHECKPOINT:?Set XLSR_CHECKPOINT to xlsr2_300m.pt}"
: "${ASVSPOOF_ROOT:?Set ASVSPOOF_ROOT to the dataset directory}"
: "${PROTOCOL_ROOT:?Set PROTOCOL_ROOT to the protocol directory}"

python train.py \
  --xlsr-checkpoint "$XLSR_CHECKPOINT" \
  --train-root "$ASVSPOOF_ROOT/ASVspoof2019_LA_train/flac" \
  --train-protocol "$PROTOCOL_ROOT/ASVspoof2019.LA.cm.train.trn.txt" \
  --dev-root "$ASVSPOOF_ROOT/ASVspoof2019_LA_dev/flac" \
  --dev-protocol "$PROTOCOL_ROOT/ASVspoof2019.LA.cm.dev.trl.txt" \
  --protocol-format asvspoof \
  --audio-extension .flac \
  --output-dir checkpoints/cmsd \
  --batch-size 4 \
  --epochs 8 \
  --learning-rate 1e-6 \
  --rawboost-algo 5
