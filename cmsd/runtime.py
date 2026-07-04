"""Shared command-line and checkpoint helpers."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .model import CMSD, CMSDConfig


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--xlsr-checkpoint", required=True)
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        help="OpenAI Whisper variant, e.g. tiny, base, small, medium, or large-v3",
    )
    parser.add_argument(
        "--whisper-cache",
        help="Directory used to cache the selected Whisper checkpoint",
    )
    parser.add_argument("--embedding-dim", type=int, default=288)
    parser.add_argument("--num-encoder-blocks", type=int, default=6)
    parser.add_argument("--asp", action="store_true")
    parser.add_argument("--freeze-backbones", action="store_true")


def build_model(args: argparse.Namespace) -> CMSD:
    return CMSD(
        CMSDConfig(
            xlsr_checkpoint=args.xlsr_checkpoint,
            whisper_model=args.whisper_model,
            whisper_cache=args.whisper_cache,
            embedding_dim=args.embedding_dim,
            num_encoder_blocks=args.num_encoder_blocks,
            attentive_statistics_pooling=args.asp,
            freeze_backbones=args.freeze_backbones,
        )
    )


def load_checkpoint(model: CMSD, path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    return checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
