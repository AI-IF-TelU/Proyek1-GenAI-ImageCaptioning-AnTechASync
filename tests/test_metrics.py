"""Uji metrik evaluasi caption tanpa model atau dataset penuh."""

import pytest

from src.metrics import compute_caption_metrics


def test_identical_captions_receive_perfect_bleu_scores():
    predictions = ["a small dog runs quickly", "two children play outside"]
    references = [
        ["a small dog runs quickly"],
        ["two children play outside"],
    ]

    metrics = compute_caption_metrics(predictions, references)

    assert metrics["bleu_1"] == pytest.approx(1.0)
    assert metrics["bleu_2"] == pytest.approx(1.0)
    assert metrics["bleu_3"] == pytest.approx(1.0)
    assert metrics["bleu_4"] == pytest.approx(1.0)
    assert 0.0 < metrics["meteor"] <= 1.0
    assert metrics["cider"] > 0.0


def test_related_caption_scores_better_than_unrelated_caption():
    references = [["a black dog runs through green grass"]]

    related = compute_caption_metrics(
        ["a black dog runs through grass"], references
    )
    unrelated = compute_caption_metrics(
        ["three people sit beside a table"], references
    )

    assert related["bleu_1"] > unrelated["bleu_1"]
    assert related["meteor"] > unrelated["meteor"]
    assert related["cider"] > unrelated["cider"]


def test_metrics_use_the_best_of_multiple_references():
    metrics = compute_caption_metrics(
        ["a puppy runs fast"],
        [["a cat sleeps indoors", "a puppy runs fast"]],
    )

    assert metrics["bleu_1"] == pytest.approx(1.0)
    assert metrics["meteor"] > 0.9


def test_metrics_reject_mismatched_or_empty_inputs():
    with pytest.raises(ValueError, match="jumlah prediksi"):
        compute_caption_metrics(["one caption"], [])

    with pytest.raises(ValueError, match="minimal satu"):
        compute_caption_metrics([], [])

    with pytest.raises(ValueError, match="reference kosong"):
        compute_caption_metrics(["one caption"], [[]])
