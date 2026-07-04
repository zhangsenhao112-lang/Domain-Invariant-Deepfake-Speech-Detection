"""CMSD network assembled from HLA, CPMF, and DP-BiVMamba."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .backbones import WhisperBackbone, XLSRBackbone
from .cpmf import CPMF
from .dp_bivmamba import MixerModel
from .pooling import HierarchicalLayerAggregation


@dataclass
class CMSDConfig:
    xlsr_checkpoint: str | Path
    whisper_model: str = "large-v3"
    whisper_cache: str | None = None
    xlsr_layers: int = 24
    embedding_dim: int = 288
    num_encoder_blocks: int = 6
    attentive_statistics_pooling: bool = False
    freeze_backbones: bool = False


class CMSD(nn.Module):
    """Collaborative Multi-view Spoofing Detector."""

    def __init__(self, config: CMSDConfig):
        super().__init__()
        self.config = config
        self.xlsr = XLSRBackbone(config.xlsr_checkpoint)
        self.whisper = WhisperBackbone(config.whisper_model, config.whisper_cache)
        self.xlsr_hla = HierarchicalLayerAggregation(config.xlsr_layers)
        self.whisper_hla = HierarchicalLayerAggregation(self.whisper.num_layers)
        self.xlsr_norm = nn.BatchNorm2d(1)
        self.whisper_norm = nn.BatchNorm2d(1)
        self.cpmf = CPMF(
            input_dims=(1024, self.whisper.output_dim),
            stage_dims=(512, 256, config.embedding_dim // 2),
        )
        self.input_norm = nn.BatchNorm2d(1)
        self.activation = nn.SELU(inplace=True)
        self.detector = MixerModel(
            d_model=config.embedding_dim,
            n_layer=config.num_encoder_blocks // 2,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            ASP=config.attentive_statistics_pooling,
            multiscale=True,
        )
        if config.freeze_backbones:
            for module in (self.xlsr, self.whisper):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    @staticmethod
    def _view_norm(x: torch.Tensor, norm: nn.BatchNorm2d) -> torch.Tensor:
        return norm(x.unsqueeze(1)).squeeze(1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        xlsr = self.xlsr_hla(self.xlsr(waveform))
        whisper = self.whisper_hla(self.whisper(waveform))
        xlsr = self._view_norm(xlsr, self.xlsr_norm)
        whisper = self._view_norm(whisper, self.whisper_norm)
        length = min(xlsr.shape[1], whisper.shape[1])
        xlsr, whisper = self.cpmf(xlsr[:, :length], whisper[:, :length])
        fused = torch.cat((xlsr, whisper), dim=-1)
        fused = self.activation(self._view_norm(fused, self.input_norm))
        return self.detector(fused)
