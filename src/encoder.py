"""
EncoderCNN: "mata" model image captioning.

Mengubah gambar (B, 3, 128, 128) menjadi vektor fitur (B, 256) yang meringkas isi gambar.
Vanilla CNN yang disusun sendiri, bobot dimulai dari acak (tanpa model pretrained).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Satu blok CNN: Conv2d(3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2).

    - Conv 3x3 + padding 1 : mencari pola lokal tanpa mengubah tinggi/lebar
    - BatchNorm            : menstabilkan nilai antar lapisan supaya training lancar
    - ReLU                 : membuang nilai negatif agar model bisa belajar pola non-linear
    - MaxPool 2            : memperkecil tinggi/lebar setengahnya, menyimpan nilai terkuat

    Conv memakai bias=False karena BatchNorm sudah punya parameter geser (beta) sendiri.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_channels, H, W) -> (B, out_channels, H/2, W/2)
        return self.block(x)


class EncoderCNN(nn.Module):
    """
    Vanilla CNN 5 blok -> Global Average Pooling -> Linear -> Dropout.

    Channel bertambah (32 -> 256) sementara ukuran gambar mengecil (128 -> 4):
    lapisan awal menangkap pola sederhana (garis, warna), lapisan dalam menangkap
    pola kompleks (bentuk, bagian objek).
    """

    def __init__(self, feature_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),     # (B, 3, 128, 128)  -> (B, 32, 64, 64)
            ConvBlock(32, 64),    # (B, 32, 64, 64)   -> (B, 64, 32, 32)
            ConvBlock(64, 128),   # (B, 64, 32, 32)   -> (B, 128, 16, 16)
            ConvBlock(128, 256),  # (B, 128, 16, 16)  -> (B, 256, 8, 8)
            ConvBlock(256, 256),  # (B, 256, 8, 8)    -> (B, 256, 4, 4)
        )

        # Global Average Pooling: rata-rata tiap channel di seluruh area 4x4.
        # Dibanding flatten (256*4*4 = 4096 angka), GAP hanya menghasilkan 256 angka,
        # sehingga Linear sesudahnya jauh lebih kecil dan model tidak mudah overfit.
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(256, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)  # (B, 256, 4, 4)
        x = self.pool(x)           # (B, 256, 1, 1)
        x = x.flatten(1)           # (B, 256)
        x = self.fc(x)             # (B, feature_dim)
        x = self.dropout(x)        # (B, feature_dim)
        return x
