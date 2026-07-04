# Third-party notices

This implementation is organized from research code built on the following
projects:

- [XLSR-Mamba](https://github.com/swagshaw/XLSR-Mamba): training pipeline and
  bidirectional detector baseline.
- [MSV-Mamba](https://github.com/YuHengsss/MSVMamba): selective-scan utilities
  adapted by `cmsd/dp_bivmamba.py`.
- [DepMamba](https://github.com/FlameSky-S/DepMamba): cross-modal bidirectional
  Mamba used by CPMF.
- [state-spaces/mamba](https://github.com/state-spaces/mamba): Mamba CUDA
  kernels and PyTorch modules.
- [fairseq](https://github.com/facebookresearch/fairseq): XLS-R checkpoint
  loading.
- [openai/whisper](https://github.com/openai/whisper): spectrogram-view
  backbone.
- [RawBoost](https://github.com/TakHemlata/RawBoost-antispoofing): waveform
  augmentation.

The vendored files retain their original copyright headers where present.
Please also follow each upstream project's license when redistributing or
modifying those components.
