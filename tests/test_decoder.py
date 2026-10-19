"""
Uji DecoderLSTM dengan tensor acak (belum butuh dataset / vocabulary asli).

Jalankan dari root repo:  python -m tests.test_decoder
"""

import torch

from src.decoder import END_IDX, PAD_IDX, START_IDX, DecoderLSTM

B, T, VOCAB = 4, 30, 1000


def make_captions(device):
    # Caption palsu: <start> kata-kata acak ... <end> <pad> <pad> ...
    caps = torch.randint(4, VOCAB, (B, T), device=device)
    caps[:, 0] = START_IDX
    caps[:, 20] = END_IDX
    caps[:, 21:] = PAD_IDX
    return caps


def test_decoder():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    decoder = DecoderLSTM(vocab_size=VOCAB).to(device)
    features = torch.randn(B, 256, device=device, requires_grad=True)
    captions = make_captions(device)

    # 1. Shape sesuai kontrak
    logits = decoder(features, captions)
    print(f"forward      : fitur {tuple(features.shape)} + caption {tuple(captions.shape)} "
          f"-> logits {tuple(logits.shape)}")
    assert logits.shape == (B, T - 1, VOCAB)

    # 2. Teacher forcing tidak "mengintip" kata masa depan:
    #    mengubah kata ke-15 dst. tidak boleh mengubah tebakan langkah 0..14
    decoder.eval()
    base = decoder(features, captions)
    changed = captions.clone()
    changed[:, 15:] = torch.randint(4, VOCAB, (B, T - 15), device=device)
    after = decoder(features, changed)
    assert torch.allclose(base[:, :15], after[:, :15]), "Langkah awal berubah: ada kebocoran kata masa depan"
    assert not torch.allclose(base[:, 15:], after[:, 15:])
    print("Teacher force: tebakan langkah t hanya bergantung pada kata 0..t (tidak mengintip)")

    # 3. Fitur gambar benar-benar dipakai: gambar beda -> tebakan beda
    other = decoder(torch.randn(B, 256, device=device), captions)
    assert not torch.allclose(base, other)
    print("Fitur gambar : gambar berbeda menghasilkan tebakan berbeda")

    # 4. Embedding <pad> selalu nol
    assert decoder.embedding.weight[PAD_IDX].abs().sum() == 0
    print("Embedding pad: vektor <pad> = 0")

    # 5. Gradien mengalir sampai fitur gambar (agar encoder ikut belajar)
    decoder.train()
    decoder(features, captions).sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    print("Gradien      : mengalir sampai fitur gambar")

    # 6. generate (greedy) untuk satu gambar
    decoder.eval()
    words = decoder.generate(features[:1].detach(), max_len=T)
    assert isinstance(words, list) and len(words) <= T - 1
    assert all(isinstance(w, int) and 0 <= w < VOCAB for w in words)
    assert START_IDX not in words and END_IDX not in words
    print(f"generate     : {len(words)} kata, contoh nomor kata {words[:8]}...")
    print("               (masih acak karena belum dilatih)")

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nJumlah parameter (vocab {VOCAB}): {n_params:,}")
    print(f"Device: {device}")
    print("LULUS: DecoderLSTM")


if __name__ == "__main__":
    test_decoder()
