"""Training lokal untuk model image captioning Flickr8k."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (
    MAX_LEN,
    SPLIT_SEED,
    DataPipeline,
    Flickr8kDataset,
    build_eval_transform,
    create_data_pipeline,
)
from src.model import CaptioningModel
from src.vocab import PAD_IDX

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Tetapkan seed Python dan PyTorch untuk eksekusi yang reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """Pilih device lokal dari ``auto``, ``cpu``, atau ``cuda``."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi tidak tersedia pada komputer ini")
    if name not in {"cpu", "cuda"}:
        raise ValueError(f"Device tidak dikenal: {name}")
    return torch.device(name)


def _loss_and_token_count(
    logits: torch.Tensor,
    captions: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, int]:
    """Hitung loss teacher forcing dan jumlah target selain ``<pad>``."""
    targets = captions[:, 1:]  # (B, 29)
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "Shape logits dan target tidak cocok: "
            f"{tuple(logits.shape)} vs {tuple(targets.shape)}"
        )

    valid_tokens = int(targets.ne(PAD_IDX).sum().item())
    if valid_tokens == 0:
        raise ValueError("Batch tidak memiliki target selain <pad>")

    vocab_size = logits.size(-1)
    loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
    return loss, valid_tokens


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Latih model satu epoch dan kembalikan loss rata-rata per token."""
    model.train()
    loss_sum = 0.0
    token_count = 0

    for images, captions in data_loader:
        images = images.to(device, non_blocking=True)  # (B, 3, 128, 128)
        captions = captions.to(device, non_blocking=True)  # (B, 30)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images, captions)  # (B, 29, V)
        loss, valid_tokens = _loss_and_token_count(logits, captions, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_sum += loss.item() * valid_tokens
        token_count += valid_tokens

    if token_count == 0:
        raise ValueError("DataLoader training kosong")
    return loss_sum / token_count


def validate_one_epoch(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    device: torch.device,
) -> float:
    """Hitung loss validation/test tanpa mengubah bobot model."""
    model.eval()
    loss_sum = 0.0
    token_count = 0

    with torch.inference_mode():
        for images, captions in data_loader:
            images = images.to(device, non_blocking=True)  # (B, 3, 128, 128)
            captions = captions.to(device, non_blocking=True)  # (B, 30)
            logits = model(images, captions)  # (B, 29, V)
            loss, valid_tokens = _loss_and_token_count(logits, captions, criterion)
            loss_sum += loss.item() * valid_tokens
            token_count += valid_tokens

    if token_count == 0:
        raise ValueError("DataLoader evaluasi kosong")
    return loss_sum / token_count


def build_checkpoint(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_val_loss: float,
    epochs_without_improvement: int,
    config: Mapping[str, Any],
    vocab_tokens: Sequence[str],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Susun checkpoint dari state dictionary dan metadata sederhana."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "config": dict(config),
        "vocab_tokens": list(vocab_tokens),
        "history": [dict(item) for item in history],
    }


def save_checkpoint(checkpoint: Mapping[str, Any], path: str | Path) -> None:
    """Simpan checkpoint secara atomik agar file lama tidak rusak."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(dict(checkpoint), temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Muat checkpoint lokal yang hanya berisi tensor dan tipe Python dasar."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {source}")
    checkpoint = torch.load(source, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Format checkpoint harus berupa dictionary")
    return checkpoint


def create_overfit_loaders(
    pipeline: DataPipeline,
    *,
    image_count: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    """Buat loader diagnostik dari sejumlah kecil gambar train yang sama."""
    if not 1 <= image_count <= len(pipeline.splits.train):
        raise ValueError(
            f"overfit-images harus 1..{len(pipeline.splits.train)}, dapat {image_count}"
        )

    image_names = pipeline.splits.train[:image_count]
    dataset = Flickr8kDataset(
        pipeline.train_dataset.images_dir,
        pipeline.captions_by_image,
        image_names,
        pipeline.vocab,
        transform=build_eval_transform(),
        max_len=pipeline.train_dataset.max_len,
    )
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(dataset, shuffle=True, generator=generator, **common)
    validation_loader = DataLoader(dataset, shuffle=False, **common)
    return train_loader, validation_loader


def save_history(history: Sequence[Mapping[str, Any]], results_dir: str | Path) -> None:
    """Simpan riwayat loss sebagai JSON dan grafik PNG."""
    destination = Path(results_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "training_history.json").write_text(
        json.dumps(list(history), indent=2), encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_losses, marker="o", label="Train loss")
    axis.plot(epochs, val_losses, marker="o", label="Validation loss")
    axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Riwayat training")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "loss_curve.png", dpi=150)
    plt.close(figure)


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Ambil konfigurasi sederhana yang aman disimpan di checkpoint."""
    return {
        "feature_dim": 256,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "min_freq": args.min_freq,
        "max_len": MAX_LEN,
        "seed": args.seed,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "overfit_images": args.overfit_images,
    }


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "learning-rate": args.learning_rate,
        "grad-clip": args.grad_clip,
        "patience": args.patience,
        "scheduler-patience": args.scheduler_patience,
        "embed-dim": args.embed_dim,
        "hidden-dim": args.hidden_dim,
        "min-freq": args.min_freq,
        "val-size": args.val_size,
        "test-size": args.test_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Nilai harus positif: {', '.join(invalid)}")
    if args.weight_decay < 0 or args.num_workers < 0 or args.overfit_images < 0:
        raise ValueError("weight-decay, num-workers, dan overfit-images tidak boleh negatif")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout harus berada pada rentang [0, 1)")
    if not 0.0 < args.scheduler_factor < 1.0:
        raise ValueError("scheduler-factor harus berada pada rentang (0, 1)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse opsi training lokal dari command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=REPO_ROOT / "data" / "Images")
    parser.add_argument(
        "--captions-file", type=Path, default=REPO_ROOT / "data" / "captions.txt"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints"
    )
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--overfit-images",
        type=int,
        default=0,
        help="Gunakan N gambar train sebagai uji hafalan; 0 berarti training penuh",
    )
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _restore_training_state(
    checkpoint: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Mapping[str, Any],
    vocab_tokens: Sequence[str],
) -> tuple[int, float, int, list[dict[str, Any]]]:
    """Pulihkan training dan pastikan data/arsitektur tetap kompatibel."""
    if list(checkpoint.get("vocab_tokens", [])) != list(vocab_tokens):
        raise ValueError("Vocabulary checkpoint tidak sama dengan vocabulary data lokal")

    saved_config = checkpoint.get("config", {})
    compatibility_keys = (
        "feature_dim",
        "embed_dim",
        "hidden_dim",
        "dropout",
        "min_freq",
        "max_len",
        "seed",
        "val_size",
        "test_size",
        "overfit_images",
    )
    mismatches = [
        key
        for key in compatibility_keys
        if saved_config.get(key) != config.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Konfigurasi resume berbeda pada: " + ", ".join(mismatches)
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint["best_val_loss"])
    epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
    history = [dict(item) for item in checkpoint.get("history", [])]
    return start_epoch, best_val_loss, epochs_without_improvement, history


def run_training(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Jalankan training lokal lengkap dan kembalikan riwayat epoch."""
    set_seed(args.seed)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    config = _config_from_args(args)

    pipeline = create_data_pipeline(
        args.images_dir,
        args.captions_file,
        batch_size=args.batch_size,
        min_freq=args.min_freq,
        max_len=MAX_LEN,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    train_loader = pipeline.train_loader
    validation_loader = pipeline.val_loader

    checkpoint_dir = args.checkpoint_dir
    results_dir = args.results_dir
    if args.overfit_images:
        train_loader, validation_loader = create_overfit_loaders(
            pipeline,
            image_count=args.overfit_images,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        checkpoint_dir = checkpoint_dir / "overfit"
        results_dir = results_dir / "overfit"

    model = CaptioningModel(
        vocab_size=len(pipeline.vocab),
        feature_dim=256,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, device)
        (
            start_epoch,
            best_val_loss,
            epochs_without_improvement,
            history,
        ) = _restore_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            vocab_tokens=pipeline.vocab.id_to_token,
        )

    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint sudah mencapai epoch {start_epoch - 1}; "
            f"target --epochs hanya {args.epochs}"
        )

    print(f"Device       : {device}")
    print(f"Vocabulary   : {len(pipeline.vocab):,} token")
    print(f"Train batch  : {len(train_loader):,}")
    print(f"Valid batch  : {len(validation_loader):,}")
    print(f"Checkpoint   : {checkpoint_dir.resolve()}")
    if args.overfit_images:
        print(f"Mode         : uji hafalan {args.overfit_images} gambar")
    else:
        print("Mode         : training penuh")

    best_path = checkpoint_dir / "best_model.pt"
    last_path = checkpoint_dir / "last_checkpoint.pt"
    for epoch in range(start_epoch, args.epochs + 1):
        started_at = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_loss = validate_one_epoch(
            model, validation_loader, criterion, device=device
        )
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        elapsed = time.perf_counter() - started_at
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "elapsed_seconds": elapsed,
        }
        history.append(epoch_record)
        checkpoint = build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
            config=config,
            vocab_tokens=pipeline.vocab.id_to_token,
            history=history,
        )
        save_checkpoint(checkpoint, last_path)
        if improved:
            save_checkpoint(checkpoint, best_path)

        marker = " | best" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"lr {current_lr:.2e} | {elapsed:.1f}s{marker}"
        )
        if epochs_without_improvement >= args.patience:
            print(
                "Early stopping: validation loss tidak membaik selama "
                f"{args.patience} epoch."
            )
            break

    save_history(history, results_dir)
    print(f"Model terbaik : {best_path.resolve()}")
    print(f"Grafik loss   : {(results_dir / 'loss_curve.png').resolve()}")
    return history


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
