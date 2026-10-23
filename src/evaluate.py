"""Evaluasi lokal checkpoint model image captioning Flickr8k."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from src.dataset import DataPipeline, build_eval_transform, create_data_pipeline
from src.metrics import compute_caption_metrics
from src.model import CaptioningModel
from src.train import load_checkpoint, resolve_device, validate_one_epoch
from src.vocab import PAD_IDX, Vocabulary

REPO_ROOT = Path(__file__).resolve().parents[1]


def prediction_record(
    *,
    image_name: str,
    predicted_ids: Sequence[int],
    references: Sequence[str],
    vocab: Vocabulary,
) -> dict[str, Any]:
    """Bentuk satu hasil prediksi yang siap disimpan sebagai JSON."""
    return {
        "image": image_name,
        "prediction": vocab.decode(predicted_ids),
        "references": list(references),
    }


def validate_checkpoint_vocab(
    checkpoint_tokens: Sequence[str], vocab: Vocabulary
) -> None:
    """Tolak evaluasi jika ID token berbeda dari saat model dilatih."""
    if list(checkpoint_tokens) != vocab.id_to_token:
        raise ValueError(
            "Vocabulary checkpoint tidak sama dengan vocabulary yang dibangun dari data train"
        )


def metrics_from_prediction_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Hitung metrik corpus dari record prediksi seluruh split evaluasi."""
    return compute_caption_metrics(
        [str(record["prediction"]) for record in records],
        [list(record["references"]) for record in records],
    )


def generate_prediction_records(
    model: CaptioningModel,
    pipeline: DataPipeline,
    *,
    device: torch.device,
    sample_count: int,
    max_len: int,
) -> list[dict[str, Any]]:
    """Generate satu caption per gambar unik dari split test."""
    transform = build_eval_transform()
    records: list[dict[str, Any]] = []
    model.eval()
    for image_name in pipeline.splits.test[:sample_count]:
        image_path = pipeline.test_dataset.images_dir / image_name
        with Image.open(image_path) as image:
            image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        predicted_ids = model.generate(image_tensor, max_len=max_len)
        records.append(
            prediction_record(
                image_name=image_name,
                predicted_ids=predicted_ids,
                references=pipeline.captions_by_image[image_name],
                vocab=pipeline.vocab,
            )
        )
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse opsi evaluasi lokal dari command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=REPO_ROOT / "data" / "Images")
    parser.add_argument(
        "--captions-file", type=Path, default=REPO_ROOT / "data" / "captions.txt"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "results" / "evaluation.json"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch-size harus positif")
    if args.num_workers < 0 or args.samples < 0:
        raise ValueError("num-workers dan samples tidak boleh negatif")
    return args


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Muat checkpoint terbaik lalu evaluasi test set lokal."""
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint tidak memiliki konfigurasi training")

    pipeline = create_data_pipeline(
        args.images_dir,
        args.captions_file,
        batch_size=args.batch_size,
        min_freq=int(config["min_freq"]),
        max_len=int(config["max_len"]),
        seed=int(config["seed"]),
        val_size=int(config["val_size"]),
        test_size=int(config["test_size"]),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validate_checkpoint_vocab(checkpoint.get("vocab_tokens", []), pipeline.vocab)

    model = CaptioningModel(
        vocab_size=len(pipeline.vocab),
        feature_dim=int(config["feature_dim"]),
        embed_dim=int(config["embed_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    test_loss = validate_one_epoch(
        model, pipeline.test_loader, criterion, device=device
    )
    all_predictions = generate_prediction_records(
        model,
        pipeline,
        device=device,
        sample_count=len(pipeline.splits.test),
        max_len=int(config["max_len"]),
    )
    metrics = metrics_from_prediction_records(all_predictions)
    samples = all_predictions[: args.samples]
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_loss": test_loss,
        "test_images": len(pipeline.splits.test),
        "metrics": metrics,
        "samples": samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Device      : {device}")
    print(f"Checkpoint  : {args.checkpoint.resolve()}")
    print(f"Test loss   : {test_loss:.4f}")
    print(
        "Metrik      : "
        f"BLEU-1 {metrics['bleu_1']:.4f} | "
        f"BLEU-2 {metrics['bleu_2']:.4f} | "
        f"BLEU-3 {metrics['bleu_3']:.4f} | "
        f"BLEU-4 {metrics['bleu_4']:.4f} | "
        f"METEOR {metrics['meteor']:.4f} | "
        f"CIDEr {metrics['cider']:.4f}"
    )
    print(f"Hasil       : {args.output.resolve()}")
    for record in samples:
        print(f"- {record['image']}: {record['prediction']}")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    run_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
