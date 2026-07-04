#!/usr/bin/env python3
"""Generate bonafide scores for any evaluation protocol."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cmsd.data import AudioTrials, read_protocol
from cmsd.runtime import add_model_arguments, build_model, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-format", choices=("simple", "asvspoof"), default="simple")
    parser.add_argument("--audio-extension", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trials = read_protocol(args.protocol, args.protocol_format)
    dataset = AudioTrials(
        args.data_root, trials, audio_extension=args.audio_extension
    )
    loader = DataLoader(
        dataset, args.batch_size, num_workers=args.workers, pin_memory=True
    )
    model = build_model(args).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch in tqdm(loader):
            waveform, identifiers = (batch[0], batch[-1])
            scores = model(waveform.to(device))[:, 1].cpu().tolist()
            for identifier, score in zip(identifiers, scores):
                handle.write(f"{identifier} {score:.10f}\n")
    print(f"Scores written to {output}")


if __name__ == "__main__":
    main()
