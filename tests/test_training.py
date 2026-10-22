"""Uji unit untuk utilitas training tanpa menjalankan Flickr8k penuh."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import (
    build_checkpoint,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    validate_one_epoch,
)
from src.vocab import PAD_IDX


class TinyCaptionModel(nn.Module):
    """Model kecil yang mengikuti kontrak keluaran ``(B, T-1, V)``."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        del images
        return self.embedding(captions[:, :-1])


def _make_loader() -> DataLoader:
    images = torch.randn(4, 3, 8, 8)
    captions = torch.tensor(
        [
            [1, 4, 5, 2, 0],
            [1, 5, 4, 2, 0],
            [1, 4, 4, 2, 0],
            [1, 5, 5, 2, 0],
        ],
        dtype=torch.int64,
    )
    return DataLoader(TensorDataset(images, captions), batch_size=2, shuffle=False)


def test_train_one_epoch_updates_weights_and_returns_finite_loss():
    model = TinyCaptionModel(vocab_size=6)
    loader = _make_loader()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = model.embedding.weight.detach().clone()

    loss = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device=torch.device("cpu"),
        grad_clip=5.0,
    )

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, model.embedding.weight.detach())
    assert model.training


def test_validate_one_epoch_does_not_update_weights():
    model = TinyCaptionModel(vocab_size=6)
    loader = _make_loader()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    before = model.embedding.weight.detach().clone()

    loss = validate_one_epoch(model, loader, criterion, device=torch.device("cpu"))

    assert torch.isfinite(torch.tensor(loss))
    assert torch.equal(before, model.embedding.weight.detach())
    assert not model.training


def test_epoch_functions_report_batch_percentage(capsys):
    model = TinyCaptionModel(vocab_size=6)
    loader = _make_loader()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device=torch.device("cpu"),
        grad_clip=5.0,
        progress_every=1,
        progress_label="Epoch 001/003 train",
    )
    validate_one_epoch(
        model,
        loader,
        criterion,
        device=torch.device("cpu"),
        progress_every=1,
        progress_label="Epoch 001/003 validasi",
    )

    output = capsys.readouterr().out
    assert "Epoch 001/003 train | Batch 2/2 (100.0%)" in output
    assert "Epoch 001/003 validasi | Batch 2/2 (100.0%)" in output
    assert "running loss" in output


def test_checkpoint_round_trip_uses_plain_state_dicts(tmp_path: Path):
    model = TinyCaptionModel(vocab_size=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    checkpoint = build_checkpoint(
        epoch=3,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        best_val_loss=1.25,
        epochs_without_improvement=2,
        config={"vocab_size": 6},
        vocab_tokens=["<pad>", "<start>", "<end>", "<unk>", "dog", "cat"],
        history=[{"epoch": 3, "train_loss": 1.5, "val_loss": 1.25}],
    )
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(checkpoint, path)
    restored = load_checkpoint(path, torch.device("cpu"))

    assert restored["epoch"] == 3
    assert restored["best_val_loss"] == 1.25
    assert restored["vocab_tokens"][-2:] == ["dog", "cat"]
    assert set(restored["model_state_dict"]) == set(model.state_dict())


def test_local_runtime_helpers_are_deterministic_and_validate_device():
    set_seed(42)
    first = torch.rand(3)
    set_seed(42)
    second = torch.rand(3)

    assert torch.equal(first, second)
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda"}
