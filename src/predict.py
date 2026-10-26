"""
Coba model pada foto sendiri: foto masuk, caption keluar.

Contoh:
    python -m src.predict --checkpoint path/best_model.pt --image foto.jpg
    python -m src.predict --checkpoint path/best_model.pt --image folder_foto/ --save hasil.png

Vocabulary diambil dari checkpoint (disimpan saat training), jadi dataset tidak dibutuhkan.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image

from src.dataset import build_eval_transform
from src.model import CaptioningModel
from src.train import load_checkpoint, resolve_device
from src.vocab import Vocabulary

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[CaptioningModel, Vocabulary, int]:
    """Bangun ulang model dan vocabulary dari checkpoint training."""
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]

    # Vocabulary persis seperti saat training (urutan ID harus sama)
    vocab = Vocabulary(min_freq=int(config["min_freq"]))
    vocab.id_to_token = list(checkpoint["vocab_tokens"])
    vocab.token_to_id = {token: index for index, token in enumerate(vocab.id_to_token)}

    model = CaptioningModel(
        vocab_size=len(vocab),
        feature_dim=int(config["feature_dim"]),
        embed_dim=int(config["embed_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab, int(config["max_len"])


def collect_images(paths: Sequence[str | Path]) -> list[Path]:
    """Terima file gambar atau folder berisi gambar."""
    images: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            images.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS))
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")
    if not images:
        raise ValueError("Tidak ada gambar yang ditemukan")
    return images


@torch.no_grad()
def caption_images(
    model: CaptioningModel, vocab: Vocabulary, images: Sequence[Path], device: torch.device, max_len: int
) -> list[str]:
    transform = build_eval_transform()  # resize 128x128 + normalisasi [-1, 1], sama seperti saat training
    captions = []
    for path in images:
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)  # (1, 3, 128, 128)
        captions.append(model.caption_image(tensor, vocab, max_len=max_len))
    return captions


def save_figure(images: Sequence[Path], captions: Sequence[str], output: Path) -> None:
    """Simpan grid gambar beserta caption hasil model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, path, caption in zip(axes.flat, images, captions):
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(caption, fontsize=10, wrap=True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buat caption untuk foto sendiri.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="File best_model.pt hasil training")
    parser.add_argument("--image", nargs="+", required=True, help="File gambar atau folder berisi gambar")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save", type=Path, help="Simpan grid gambar + caption ke file PNG")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    model, vocab, max_len = load_model(args.checkpoint, device)
    images = collect_images(args.image)
    captions = caption_images(model, vocab, images, device, max_len)

    for path, caption in zip(images, captions):
        print(f"{path.name}: {caption}")
    if args.save:
        save_figure(images, captions, args.save)
        print(f"Gambar hasil disimpan: {args.save.resolve()}")


if __name__ == "__main__":
    main()
