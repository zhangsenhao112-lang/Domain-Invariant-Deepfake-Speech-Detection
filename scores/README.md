# Score file layout

Precomputed paper scores are intentionally not distributed. Generate scores
from your own checkpoint with `generate_scores.py`, then arrange files as:

```text
scores/
├── ASVspoof2021_LA/my_model.txt
├── FoR/my_model.txt
├── In-the-Wild/my_model.txt
├── SONAR/my_model.txt
├── CodecFake/my_model.txt
└── ADD2023/my_model.txt
```

Each line is:

```text
trial_id bonafide_logit
```

Higher values indicate bonafide speech. Trial IDs must exactly match
`protocols/<dataset>/eval.txt`. Evaluate a complete set using the EER threshold
derived from the full ASVspoof 2021 LA evaluation partition:

```bash
python evaluate.py --score-name my_model.txt
```

The old A19-only threshold can be audited with `--la-subset a19`. The default
`--metric-mode paper` retains the historical Macro-F1 convention.
For class-wise Macro-F1 with consistent units, run:

```bash
python evaluate.py --score-name my_model.txt --metric-mode corrected
```
