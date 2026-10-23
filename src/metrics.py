"""Metrik caption tanpa dependensi NLP eksternal.

BLEU dihitung pada level corpus. METEOR memakai exact-token matching (tanpa
stemming/sinonim), sedangkan CIDEr memakai TF-IDF n-gram dengan IDF smoothing.
Semua teks memakai tokenizer yang sama dengan vocabulary model.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from src.vocab import tokenize

TokenSequence = Sequence[str]
Ngram = tuple[str, ...]


def _ngrams(tokens: TokenSequence, order: int) -> Counter[Ngram]:
    return Counter(
        tuple(tokens[index : index + order])
        for index in range(len(tokens) - order + 1)
    )


def _closest_reference_length(candidate_length: int, references: Sequence[TokenSequence]) -> int:
    return min(
        (len(reference) for reference in references),
        key=lambda length: (abs(length - candidate_length), length),
    )


def _corpus_bleu(
    predictions: Sequence[TokenSequence],
    references: Sequence[Sequence[TokenSequence]],
) -> dict[str, float]:
    clipped = [0] * 4
    totals = [0] * 4
    candidate_length = 0
    reference_length = 0

    for prediction, image_references in zip(predictions, references, strict=True):
        candidate_length += len(prediction)
        reference_length += _closest_reference_length(len(prediction), image_references)
        for order in range(1, 5):
            candidate_counts = _ngrams(prediction, order)
            totals[order - 1] += sum(candidate_counts.values())
            maximum_reference_counts: Counter[Ngram] = Counter()
            for reference in image_references:
                reference_counts = _ngrams(reference, order)
                maximum_reference_counts |= reference_counts
            clipped[order - 1] += sum(
                min(count, maximum_reference_counts[ngram])
                for ngram, count in candidate_counts.items()
            )

    if candidate_length == 0:
        return {f"bleu_{order}": 0.0 for order in range(1, 5)}

    brevity_penalty = (
        1.0
        if candidate_length > reference_length
        else math.exp(1.0 - reference_length / candidate_length)
    )
    precisions = [
        numerator / denominator if denominator else 0.0
        for numerator, denominator in zip(clipped, totals, strict=True)
    ]
    scores: dict[str, float] = {}
    for order in range(1, 5):
        selected = precisions[:order]
        if any(precision == 0.0 for precision in selected):
            score = 0.0
        else:
            score = brevity_penalty * math.exp(
                sum(math.log(precision) for precision in selected) / order
            )
        scores[f"bleu_{order}"] = score
    return scores


def _meteor_single(prediction: TokenSequence, reference: TokenSequence) -> float:
    if not prediction or not reference:
        return 0.0

    unused_positions: dict[str, list[int]] = {}
    for index, token in enumerate(reference):
        unused_positions.setdefault(token, []).append(index)

    aligned_positions: list[int] = []
    for token in prediction:
        positions = unused_positions.get(token)
        if positions:
            aligned_positions.append(positions.pop(0))

    matches = len(aligned_positions)
    if matches == 0:
        return 0.0
    precision = matches / len(prediction)
    recall = matches / len(reference)
    harmonic_mean = (10.0 * precision * recall) / (recall + 9.0 * precision)
    chunks = 1 + sum(
        current != previous + 1
        for previous, current in zip(aligned_positions, aligned_positions[1:])
    )
    penalty = 0.5 * (chunks / matches) ** 3
    return harmonic_mean * (1.0 - penalty)


def _meteor(
    predictions: Sequence[TokenSequence],
    references: Sequence[Sequence[TokenSequence]],
) -> float:
    return sum(
        max(_meteor_single(prediction, reference) for reference in image_references)
        for prediction, image_references in zip(predictions, references, strict=True)
    ) / len(predictions)


def _cosine_similarity(
    left: dict[Ngram, float], right: dict[Ngram, float]
) -> float:
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot_product = sum(value * right.get(ngram, 0.0) for ngram, value in left.items())
    return dot_product / (left_norm * right_norm)


def _cider(
    predictions: Sequence[TokenSequence],
    references: Sequence[Sequence[TokenSequence]],
) -> float:
    image_count = len(predictions)
    document_frequency: list[Counter[Ngram]] = [Counter() for _ in range(4)]
    for image_references in references:
        for order in range(1, 5):
            unique_ngrams = set().union(
                *(_ngrams(reference, order) for reference in image_references)
            )
            document_frequency[order - 1].update(unique_ngrams)

    def vector(tokens: TokenSequence, order: int) -> dict[Ngram, float]:
        counts = _ngrams(tokens, order)
        total = sum(counts.values())
        if total == 0:
            return {}
        return {
            ngram: (count / total)
            * (math.log((image_count + 1) / (document_frequency[order - 1][ngram] + 1)) + 1.0)
            for ngram, count in counts.items()
        }

    image_scores: list[float] = []
    for prediction, image_references in zip(predictions, references, strict=True):
        order_scores: list[float] = []
        for order in range(1, 5):
            prediction_vector = vector(prediction, order)
            reference_scores = []
            for reference in image_references:
                length_penalty = math.exp(
                    -((len(prediction) - len(reference)) ** 2) / (2.0 * 6.0**2)
                )
                reference_scores.append(
                    _cosine_similarity(prediction_vector, vector(reference, order))
                    * length_penalty
                )
            order_scores.append(sum(reference_scores) / len(reference_scores))
        image_scores.append(10.0 * sum(order_scores) / 4.0)
    return sum(image_scores) / len(image_scores)


def compute_caption_metrics(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> dict[str, float]:
    """Hitung BLEU-1..4, METEOR, dan CIDEr untuk sekumpulan gambar."""
    if len(predictions) != len(references):
        raise ValueError("jumlah prediksi dan reference harus sama")
    if not predictions:
        raise ValueError("evaluasi membutuhkan minimal satu prediksi")
    if any(not image_references for image_references in references):
        raise ValueError("setiap gambar tidak boleh memiliki reference kosong")

    tokenized_predictions = [tokenize(prediction) for prediction in predictions]
    tokenized_references = [
        [tokenize(reference) for reference in image_references]
        for image_references in references
    ]
    return {
        **_corpus_bleu(tokenized_predictions, tokenized_references),
        "meteor": _meteor(tokenized_predictions, tokenized_references),
        "cider": _cider(tokenized_predictions, tokenized_references),
    }
