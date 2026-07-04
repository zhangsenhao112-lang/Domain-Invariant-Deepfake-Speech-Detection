"""Collaborative Progressive Multi-view Fusion (CPMF)."""

from dataclasses import dataclass

import torch
from torch import nn

from .third_party.cobi_mamba.mm_bimamba import Mamba as CrossBiMamba


@dataclass
class CrossMambaConfig:
    d_state: int = 16
    expand: int = 4
    d_conv: int = 4


class ResidualProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self.skip = (
            nn.Conv1d(input_dim, output_dim, 1, bias=False)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return (self.main(x) + self.skip(x)).transpose(1, 2)


class CollaborativeStage(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, output_dim: int, cfg: CrossMambaConfig):
        super().__init__()
        self.x_projection = ResidualProjection(x_dim, output_dim)
        self.y_projection = ResidualProjection(y_dim, output_dim)
        self.mamba = CrossBiMamba(
            d_model=output_dim,
            bimamba_type="v2",
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
        )
        self.x_norm = nn.LayerNorm(output_dim, eps=1e-6)
        self.y_norm = nn.LayerNorm(output_dim, eps=1e-6)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x_projection(x)
        y = self.y_projection(y)
        x_delta, y_delta = self.mamba(x, y, None, None)
        return x + self.x_norm(x_delta), y + self.y_norm(y_delta)


class CPMF(nn.Module):
    def __init__(
        self,
        input_dims: tuple[int, int] = (1024, 1280),
        stage_dims: tuple[int, ...] = (512, 256, 144),
        mamba: CrossMambaConfig | None = None,
    ):
        super().__init__()
        cfg = mamba or CrossMambaConfig()
        x_dim, y_dim = input_dims
        stages = []
        for output_dim in stage_dims:
            stages.append(CollaborativeStage(x_dim, y_dim, output_dim, cfg))
            x_dim = y_dim = output_dim
        self.stages = nn.ModuleList(stages)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for stage in self.stages:
            x, y = stage(x, y)
        return x, y
