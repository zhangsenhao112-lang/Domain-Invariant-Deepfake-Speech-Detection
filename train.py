#!/usr/bin/env python3
"""Train CMSD on a source-domain protocol."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cmsd.data import AudioTrials, read_protocol
from cmsd.runtime import add_model_arguments, build_model, load_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--train-protocol", required=True)
    parser.add_argument("--dev-root", required=True)
    parser.add_argument("--dev-protocol", required=True)
    parser.add_argument("--protocol-format", choices=("simple", "asvspoof"), default="simple")
    parser.add_argument("--audio-extension", default="")
    parser.add_argument("--output-dir", default="checkpoints/cmsd")
    parser.add_argument("--resume")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rawboost-algo", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for waveform, label, _ in tqdm(loader, leave=False):
            waveform, label = waveform.to(device), label.to(device)
            logits = model(waveform)
            loss = criterion(logits, label)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * waveform.shape[0]
            total_items += waveform.shape[0]
    return total_loss / total_items


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_set = AudioTrials(
        args.train_root,
        read_protocol(args.train_protocol, args.protocol_format),
        rawboost_algo=args.rawboost_algo,
        audio_extension=args.audio_extension,
    )
    dev_set = AudioTrials(
        args.dev_root,
        read_protocol(args.dev_protocol, args.protocol_format),
        audio_extension=args.audio_extension,
    )
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_set, args.batch_size * 2, num_workers=args.workers, pin_memory=True
    )
    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch, best_loss = 0, float("inf")
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_loss = checkpoint.get("best_loss", best_loss)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([0.1, 0.9], device=device)
    )
    for epoch in range(start_epoch, args.epochs):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        dev_loss = run_epoch(model, dev_loader, criterion, device)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": min(best_loss, dev_loss),
            "args": vars(args),
        }
        torch.save(state, output_dir / "last.pt")
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(state, output_dir / "best.pt")
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} dev_loss={dev_loss:.6f}")


if __name__ == "__main__":
    main()
