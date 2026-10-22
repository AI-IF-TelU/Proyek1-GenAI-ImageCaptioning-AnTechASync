"""Uji unit keluaran evaluasi tanpa checkpoint atau Flickr8k penuh."""

import torch

import pytest

from src.evaluate import prediction_record, validate_checkpoint_vocab
from src.vocab import Vocabulary


def test_prediction_record_decodes_ids_and_preserves_references():
    vocab = Vocabulary(min_freq=1)
    vocab.build(["a dog runs", "a cat sleeps"])
    predicted_ids = [
        vocab.token_to_id["a"],
        vocab.token_to_id["dog"],
        vocab.token_to_id["runs"],
    ]

    record = prediction_record(
        image_name="example.jpg",
        predicted_ids=predicted_ids,
        references=["A dog runs.", "A dog is outside."],
        vocab=vocab,
    )

    assert record == {
        "image": "example.jpg",
        "prediction": "a dog runs",
        "references": ["A dog runs.", "A dog is outside."],
    }
    assert all(isinstance(index, int) for index in predicted_ids)
    assert torch.tensor(predicted_ids).dtype == torch.int64


def test_checkpoint_vocabulary_must_match_rebuilt_training_vocabulary():
    vocab = Vocabulary(min_freq=1)
    vocab.build(["a dog runs"])

    validate_checkpoint_vocab(vocab.id_to_token, vocab)

    with pytest.raises(ValueError, match="Vocabulary checkpoint"):
        validate_checkpoint_vocab(["<pad>", "<start>", "<end>", "<unk>"], vocab)
