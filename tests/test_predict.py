"""Uji src/predict.py: checkpoint palsu kecil -> caption untuk gambar baru."""

import torch
from PIL import Image

from src.model import CaptioningModel
from src.predict import caption_images, collect_images, load_model
from src.vocab import SPECIAL_TOKENS


def test_predict_from_checkpoint(tmp_path):
    tokens = [*SPECIAL_TOKENS, "a", "dog", "runs"]
    config = {"feature_dim": 256, "embed_dim": 256, "hidden_dim": 512, "dropout": 0.3,
              "min_freq": 1, "max_len": 30}
    model = CaptioningModel(vocab_size=len(tokens))
    checkpoint = tmp_path / "best_model.pt"
    torch.save({"config": config, "vocab_tokens": tokens,
                "model_state_dict": model.state_dict()}, checkpoint)

    Image.new("RGB", (200, 150), (120, 80, 40)).save(tmp_path / "foto.jpg")

    loaded, vocab, max_len = load_model(checkpoint, torch.device("cpu"))
    images = collect_images([tmp_path])
    captions = caption_images(loaded, vocab, images, torch.device("cpu"), max_len)

    assert len(vocab) == len(tokens) and vocab.token_to_id["dog"] == 5
    assert [p.name for p in images] == ["foto.jpg"]
    assert isinstance(captions[0], str)
    assert all(word in tokens for word in captions[0].split())
