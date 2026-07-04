# Domain-Invariant Representation Learning for Generalizable Deepfake Speech Detection

Clean PyTorch implementation of our Collaborative Multi-view Spoofing Detector
(CMSD). The model learns complementary waveform and spectrogram representations
with:

1. **HLA** — Hierarchical Layer Aggregation over all XLS-R and Whisper encoder
   layers.
2. **CPMF** — Collaborative Progressive Multi-view Fusion with cross-view
   bidirectional Mamba blocks.
3. **DP-BiVMamba** — a dual-path bidirectional utterance-level detector.

This directory was extracted from our experimental `XLSR-Mamba-main` workspace.
It intentionally excludes checkpoints, datasets, score dumps, plots, competing
baselines, and machine-specific paths.

## Repository layout

```text
.
├── cmsd/
│   ├── backbones.py       # XLS-R and Whisper layer features
│   ├── pooling.py         # HLA and attentive statistics pooling
│   ├── cpmf.py            # collaborative multi-view fusion
│   ├── dp_bivmamba.py     # final bidirectional detector
│   ├── model.py           # complete CMSD model
│   ├── data.py            # portable protocol/audio loader + RawBoost
│   └── third_party/       # minimal selective-scan components
├── train.py
├── generate_scores.py      # checkpoint inference and score generation
├── evaluate.py             # 21LA threshold + cross-dataset paper evaluation
├── multi_evaluate.py       # backward-compatible evaluation alias
├── protocols/              # original labels + normalized protocols
├── scores/                 # local generated scores (not distributed)
├── scripts/
└── requirements.txt
```

## Environment

The reproduced environment uses Python 3.10, PyTorch 2.2.1, CUDA 11.8,
Mamba-SSM 1.1.4, and causal-conv1d 1.1.3. A CUDA-capable Linux environment is
required by the selective-scan kernels.

```bash
conda create -n cmsd python=3.10 -y
conda activate cmsd
python -m pip install "pip<24.1" setuptools wheel ninja
pip install --no-build-isolation -r requirements.txt
```

The `pip<24.1` pin is needed because the fairseq commit used by XLS-R contains
legacy dependency metadata rejected by newer pip releases.

If `mamba-ssm` compilation fails, verify that `nvcc --version` is compatible
with the CUDA version reported by `python -c "import torch; print(torch.version.cuda)"`.

Download the XLS-R checkpoint:

```bash
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt
```

Whisper `large-v3` is the default and is downloaded automatically on first
use. The implementation also supports other OpenAI Whisper variants:

```bash
python train.py ... --whisper-model medium
python generate_scores.py ... --whisper-model medium
```

Common choices are `tiny`, `base`, `small`, `medium`, and `large-v3`. The code
automatically reads the selected encoder's layer count and hidden dimension,
then adjusts HLA and the first CPMF projection:

- variant loading and dimension discovery:
  [`cmsd/backbones.py`](cmsd/backbones.py), `WhisperBackbone`;
- default variant:
  [`cmsd/model.py`](cmsd/model.py), `CMSDConfig.whisper_model`;
- CLI option:
  [`cmsd/runtime.py`](cmsd/runtime.py), `--whisper-model`;
- HLA/CPMF adaptation:
  [`cmsd/model.py`](cmsd/model.py), `CMSD.__init__`.

Changing the Whisper variant changes model parameter shapes. Train a separate
checkpoint for each variant and pass the same `--whisper-model` value during
score generation. A `large-v3` checkpoint cannot be loaded into `medium`,
`small`, or another variant. Use `--whisper-cache /path/to/cache` to select the
download directory.

## Dataset downloads

The following are the original project or archive links for every dataset used
in the paper. Please read and comply with the license on each download page;
the datasets are not redistributed by this repository.

| Dataset | Download | Partition used here | Notes |
|---|---|---|---|
| ASVspoof 2019 | [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336) ([challenge page](https://www.asvspoof.org/index2019.html)) | LA train and development | Download the LA speech archives and CM protocols. Training uses this source domain. |
| ASVspoof 2021 | [LA speech](https://zenodo.org/records/4837263), [DF speech](https://zenodo.org/records/4835108), [official keys and metadata](https://www.asvspoof.org/index2021.html) | LA evaluation; DF can be evaluated in the same way | ASVspoof 2021 provides evaluation data only and reuses the 2019 train/dev sets. The official page also links PA if required. |
| ADD 2023 | [Track 1.2 train/dev](https://zenodo.org/records/12151404), [Track 1.2 evaluation Round 2](https://zenodo.org/records/12176326), [challenge site](http://addchallenge.cn/add2023) | Track 1.2, test Round 2 | This is the exact `Track1.2/testR2` split used by the experimental code. It is released under CC BY-NC-ND 4.0. |
| Fake-or-Real (FoR) | [official York University page](https://bil.eecs.yorku.ca/datasets/), [direct `for-original` archive](https://bil.eecs.yorku.ca/share/for-original.tar.gz) | `for-original/testing` | The official page also provides `for-norm`, `for-2sec`, and `for-rerec`. |
| In-the-Wild | [official project page](https://deepfake-total.com/in_the_wild), [Hugging Face download](https://huggingface.co/datasets/mueller91/In-The-Wild) | Full evaluation set | The Hugging Face repository provides `release_in_the_wild.zip` together with `meta.csv`. |
| CodecFake | [official project page](https://codecfake.github.io/), [Hugging Face dataset](https://huggingface.co/datasets/rogertseng/CodecFake), [code](https://github.com/roger-tseng/CodecFake) | Codec groups C1--C6 | This refers to the original CodecFake release, not the later CodecFake+ dataset. |
| SONAR | [official repository](https://github.com/Jessegator/SONAR), [official Google Drive data](https://drive.google.com/drive/folders/1kSqjuHiElNigCvGxD6sVKiyVaXA3xO5A?usp=sharing) | Full SONAR evaluation set | The dataset contains real speech and samples from nine synthesis sources. Keep the source folder names such as `AudioGen`, `FlashSpeech`, and `real`. |

Useful download commands:

```bash
# In-the-Wild
wget https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip
unzip release_in_the_wild.zip

# FoR variant used in the paper
wget https://bil.eecs.yorku.ca/share/for-original.tar.gz
tar -xzf for-original.tar.gz

# CodecFake (requires: pip install -U huggingface_hub)
huggingface-cli download rogertseng/CodecFake \
  --repo-type dataset --local-dir CodecFake
```

ADD 2023 Track 1.2 Round 2 is stored as multiple
`track1_2-testR2.tar.gz.00`--`.07` files. Download every part from the same
Zenodo record, then join and extract them:

```bash
cat track1_2-testR2.tar.gz.* > track1_2-testR2.tar.gz
tar -xzf track1_2-testR2.tar.gz
```

For reproducible comparisons, do not silently substitute similarly named
datasets: use ADD 2023 **Track 1.2 Round 2**, the original **CodecFake** release,
and the SONAR dataset linked from its official repository.

## Data protocols

The generic format is one trial per line:

```text
relative/path/to/audio.wav bonafide
relative/path/to/spoof.flac spoof
```

Labels may be `bonafide`, `real`, `spoof`, or `fake`. An unlabeled protocol
containing only paths is accepted by `generate_scores.py`. Standard five-column
ASVspoof protocols are enabled with `--protocol-format asvspoof`. Paths in a
protocol are resolved relative to the corresponding `--train-root`,
`--dev-root`, or `--data-root`.

For ASVspoof protocols, the utterance ID has no extension; pass
`--audio-extension .flac`. The provided scripts already do this.

## Training

```bash
python train.py \
  --xlsr-checkpoint /path/to/xlsr2_300m.pt \
  --train-root /path/to/train/audio \
  --train-protocol /path/to/train.txt \
  --dev-root /path/to/dev/audio \
  --dev-protocol /path/to/dev.txt \
  --output-dir checkpoints/cmsd \
  --batch-size 4 \
  --epochs 8 \
  --learning-rate 1e-6 \
  --rawboost-algo 5
```

Use `--asp` for the attentive-statistics-pooling classifier variant and
`--freeze-backbones` to train without updating XLS-R and Whisper. The default
matches the attention-pooling, fully fine-tuned experiment. `last.pt` supports
resuming with `--resume`; the lowest-development-loss model is saved as
`best.pt`.

The convenience ASVspoof script reads paths from environment variables:

```bash
export XLSR_CHECKPOINT=/path/to/xlsr2_300m.pt
export ASVSPOOF_ROOT=/path/to/asvspoof
export PROTOCOL_ROOT=/path/to/protocols
bash scripts/train_asvspoof19.sh
```

## Evaluation

```bash
python generate_scores.py \
  --xlsr-checkpoint /path/to/xlsr2_300m.pt \
  --checkpoint checkpoints/cmsd/best.pt \
  --data-root /path/to/evaluation/audio \
  --protocol /path/to/evaluation.txt \
  --output scores/evaluation.txt
```

Each output line is `<trial-path> <bonafide-logit>`. Use the official
dataset-specific evaluation package to compute EER, min t-DCF, or fixed-threshold
metrics.

For a quick EER and binary-metric check:

```bash
python metrics.py \
  --scores scores/evaluation.txt \
  --protocol /path/to/labeled-evaluation.txt
```

`metrics.py --threshold VALUE` can evaluate any independently chosen operating
point. The multi-dataset protocol below specifically derives its fixed
threshold from the complete ASVspoof 2021 LA evaluation set, as used for this
project's reported cross-dataset comparison. Use the official ASVspoof package
when reporting min t-DCF.

## Evaluation protocols and scoring code

The repository includes original evaluation annotations/file lists and
normalized `path label` protocols under [`protocols/`](protocols/README.md).
Precomputed scores are not included. Generate one identically named score file
for each dataset using the layout in [`scores/README.md`](scores/README.md),
then run the portable replacement for the experimental `evaluate/evaluate.py`:

```bash
python evaluate.py --score-name my_model.txt
```

The evaluation protocol:

1. uses the complete ASVspoof 2021 LA evaluation set;
2. computes its EER threshold without using any target-domain labels;
3. applies that fixed threshold to In-the-Wild, FoR, SONAR, CodecFake, and ADD
   2023;
4. reports EER, AUC, FAR, FRR, Accuracy, F1, Recall, Macro-F1, OOD averages,
   SONAR generator results, and CodecFake codec results.

The old `la_evaluate.py` accidentally restricted threshold estimation to
`A19 + bonafide`. That behavior is available only for auditing with
`--la-subset a19`; it is not the default. Likewise,
`--legacy-threshold-rounding` reproduces its two-decimal subprocess handoff.

By default, `--metric-mode paper` preserves the historical Macro-F1 display.
The old FoR, SONAR, and CodecFake scripts mixed percentage and fractional units
when calculating Macro-F1. Use the mathematically corrected version for new
experiments:

```bash
python evaluate.py --score-name my_model.txt --metric-mode corrected
```
