"""
Uji EncoderCNN dengan tensor acak (belum butuh dataset).

Jalankan dari root repo:  python -m tests.test_encoder
"""

import torch

from src.encoder import EncoderCNN


def test_encoder():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = EncoderCNN().to(device)
    images = torch.randn(4, 3, 128, 128, device=device)  # 4 gambar palsu

    # Cek shape setiap blok
    print("Shape tiap blok:")
    x = images
    print(f"  input   : {tuple(x.shape)}")
    for i, block in enumerate(encoder.features, start=1):
        x = block(x)
        print(f"  blok {i}  : {tuple(x.shape)}")

    # Cek keluaran akhir sesuai kontrak
    features = encoder(images)
    print(f"  keluaran: {tuple(features.shape)}")
    assert features.shape == (4, 256), f"Shape salah: {tuple(features.shape)}, harus (4, 256)"

    # Cek gradien bisa mengalir sampai lapisan pertama (syarat model bisa belajar)
    features.sum().backward()
    first_conv = encoder.features[0].block[0]
    assert first_conv.weight.grad is not None, "Gradien tidak sampai ke lapisan pertama"

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nJumlah parameter: {n_params:,}")
    print(f"Device: {device}")
    print("LULUS: EncoderCNN (4, 3, 128, 128) -> (4, 256)")


if __name__ == "__main__":
    test_encoder()
