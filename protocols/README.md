# Evaluation protocols

Each dataset directory contains `eval.txt` in the repository-wide format:

```text
relative/audio/path bonafide
relative/audio/path spoof
```

It also contains the original annotation or file-list artifact used by the
experimental XLSR-Mamba workspace:

| Directory | Original artifact | Label interpretation |
|---|---|---|
| `ASVspoof2021_LA` | `trial_metadata.txt` | utterance ID in column 2; label in column 6 |
| `FoR` | `filelist.txt` | `real/` is bonafide; `fake/` is spoof |
| `In-the-Wild` | `meta.csv` | filename in column 1; label in column 3 |
| `SONAR` | `filelist.txt` | `real_samples/` is bonafide; all generator directories are spoof |
| `CodecFake` | `labels.txt` | path and `real`/`fake` label in columns 1--2 |
| `ADD2023` | `labels.txt` | path and `genuine`/`fake` label |

The standardized files preserve the original trial order. Their IDs were
checked line-by-line against both released CMSD score variants.

These artifacts remain governed by their source dataset licenses; the
repository MIT license does not replace those terms. In particular, consult
the ASVspoof ODC-By terms, In-the-Wild license, CodecFake/VCTK attribution
requirements, and ADD 2023 CC BY-NC-ND 4.0 terms before redistribution.
