"""Waveform- and spectrogram-view SSL feature extractors."""

from pathlib import Path

import torch
from torch import nn


class XLSRBackbone(nn.Module):
    def __init__(self, checkpoint: str | Path):
        super().__init__()
        try:
            import fairseq
        except ImportError as exc:
            raise ImportError(
                "fairseq is required for XLS-R; follow the pinned install command in README.md"
            ) from exc
        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [str(checkpoint)]
        )
        self.model = models[0]

    def forward(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        result = self.model(waveform, mask=False, features_only=True)
        # fairseq layers are [T, B, D]; CMSD consistently uses [B, T, D].
        return [layer.transpose(0, 1) for layer, _ in result["layer_results"]]


class WhisperBackbone(nn.Module):
    def __init__(self, model_name: str = "large-v3", download_root: str | None = None):
        super().__init__()
        try:
            import whisper
        except ImportError as exc:
            raise ImportError("Install openai-whisper from requirements.txt") from exc
        full_model = whisper.load_model(
            model_name, device="cpu", download_root=download_root
        )
        self.encoder = full_model.encoder
        self.n_mels = full_model.dims.n_mels
        self.num_layers = full_model.dims.n_audio_layer
        self.output_dim = full_model.dims.n_audio_state

    def forward(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        import whisper

        mel = whisper.log_mel_spectrogram(waveform, n_mels=self.n_mels)
        x = torch.nn.functional.gelu(self.encoder.conv1(mel))
        x = torch.nn.functional.gelu(self.encoder.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + self.encoder.positional_embedding[: x.shape[1]]).to(x.dtype)
        layers = []
        for block in self.encoder.blocks:
            x = block(x)
            layers.append(x)
        return layers
