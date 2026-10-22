"""
CaptioningModel: gabungan EncoderCNN ("mata") dan DecoderLSTM ("mulut").

    gambar (B, 3, 128, 128) -> EncoderCNN -> fitur (B, 256) -> DecoderLSTM -> logits (B, 29, vocab_size)

Inilah model utuh yang nanti dilatih oleh train.py (Anggota 3).
"""

import torch
import torch.nn as nn

from src.decoder import DecoderLSTM
from src.encoder import EncoderCNN


class CaptioningModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 256,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = EncoderCNN(feature_dim=feature_dim, dropout=dropout)
        self.decoder = DecoderLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            dropout=dropout,
        )

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Dipakai saat training (teacher forcing).

        images  : (B, 3, 128, 128)
        captions: (B, 30)
        return  : logits (B, 29, vocab_size)

        Encoder dan decoder dilatih BERSAMA dalam satu model: gradien dari loss mengalir
        lewat decoder lalu ke encoder, sehingga encoder belajar fitur yang berguna untuk
        menulis caption.
        """
        features = self.encoder(images)             # (B, 256)
        return self.decoder(features, captions)     # (B, 29, vocab_size)

    @torch.no_grad()
    def generate(self, image: torch.Tensor, max_len: int = 30) -> list[int]:
        """
        Membuat caption untuk SATU gambar, hasilnya nomor kata.

        image : (1, 3, 128, 128)
        return: daftar nomor kata (tanpa <start>/<end>)
        """
        was_training = self.training
        self.eval()                                 # matikan Dropout, BatchNorm pakai statistik training
        features = self.encoder(image)              # (1, 256)
        words = self.decoder.generate(features, max_len=max_len)
        self.train(was_training)                    # kembalikan mode semula
        return words

    def caption_image(self, image: torch.Tensor, vocab, max_len: int = 30) -> str:
        """
        Membuat caption untuk SATU gambar, hasilnya kalimat.

        image: (1, 3, 128, 128)
        vocab: objek Vocabulary dari Anggota 2, wajib punya method decode(list[int]) -> str
        """
        return vocab.decode(self.generate(image, max_len=max_len))
