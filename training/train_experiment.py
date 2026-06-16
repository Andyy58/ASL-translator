#!/usr/bin/env python3
"""
Train an ASL feature classifier with the cleaned top-word subset.

Example:
    python training/train_experiment.py --data-root wlasl1000_top100_features
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ASLFeatureDataset
from model import ASLSequentialProcessor


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(dataset: ASLFeatureDataset) -> torch.Tensor:
    counts = torch.bincount(
        torch.tensor(dataset.labels), minlength=len(dataset.classes)
    ).float()
    weights = torch.zeros_like(counts)
    nonzero = counts > 0
    weights[nonzero] = torch.sqrt(counts[nonzero].mean() / counts[nonzero])
    return weights


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels, lengths in loader:
        features = features.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(features, lengths)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples * 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("wlasl1000_top100_features")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weighted-loss", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--scale-range", type=float, default=0.08)
    parser.add_argument("--hand-drop-prob", type=float, default=0.03)
    parser.add_argument("--frame-drop-prob", type=float, default=0.02)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_root = args.data_root / "train"
    val_root = args.data_root / "val"

    train_dataset = ASLFeatureDataset(
        root_dir=train_root,
        max_frames=args.max_frames,
        trim_zero_frames=True,
        normalize_hands=True,
        sample_strategy="uniform",
        augment=not args.no_augment,
        noise_std=args.noise_std,
        scale_range=args.scale_range,
        hand_drop_prob=args.hand_drop_prob,
        frame_drop_prob=args.frame_drop_prob,
    )
    val_dataset = ASLFeatureDataset(
        root_dir=val_root,
        max_frames=args.max_frames,
        trim_zero_frames=True,
        normalize_hands=True,
        sample_strategy="uniform",
    )

    if train_dataset.classes != val_dataset.classes:
        raise SystemExit("Train and validation class orders differ.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    device = choose_device()
    model = ASLSequentialProcessor(
        input_size=126,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=len(train_dataset.classes),
        dropout=args.dropout,
    ).to(device)

    if args.weighted_loss:
        class_weights = compute_class_weights(train_dataset).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=6,
    )

    run_name = (
        args.run_name or f"asl_top100_clean_{dt.datetime.now():%Y-%m-%d_%H-%M-%S}"
    )
    log_dir = Path("training/runs") / run_name
    checkpoint_dir = Path("training/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir)
    best_val_acc = 0.0
    best_path = checkpoint_dir / f"{run_name}_best.pt"

    print(f"Device: {device}")
    print(f"Classes: {len(train_dataset.classes)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Logging to: {log_dir}")

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step(val_acc)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalars(
            "Accuracy", {"Train": train_acc, "Validation": val_acc}, epoch
        )
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                    "args": vars(args),
                    "best_val_acc": best_val_acc,
                },
                best_path,
            )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.2f}% | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.2f}% | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

    writer.close()
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
