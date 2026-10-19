"""
Uji CaptioningModel (encoder + decoder digabung) dengan data acak.

Termasuk "uji hafalan": model dilatih sebentar pada 4 pasangan gambar-caption acak.
Jika semua komponen tersambung benar, loss harus turun drastis (model bisa menghafal).
Jika loss tidak turun, berarti ada bug di aliran data atau gradien.

Jalankan dari root repo:  python -m tests.test_model
"""

import math
import time

import torch
import torch.nn as nn

from src.decoder import END_IDX, PAD_IDX, START_IDX
from src.model import CaptioningModel

B, T, VOCAB = 4, 30, 1000


class DummyVocab:
    """Pengganti sementara Vocabulary milik Anggota 2: nomor kata -> 'w<nomor>'."""

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"w{i}" for i in ids)


def make_captions(device):
    caps = torch.randint(4, VOCAB, (B, T), device=device)
    caps[:, 0] = START_IDX
    caps[:, 12] = END_IDX
    caps[:, 13:] = PAD_IDX
    return caps


def test_model():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CaptioningModel(vocab_size=VOCAB).to(device)
    images = torch.randn(B, 3, 128, 128, device=device)
    captions = make_captions(device)

    # 1. Shape ujung ke ujung
    logits = model(images, captions)
    print(f"forward : gambar {tuple(images.shape)} + caption {tuple(captions.shape)} "
          f"-> logits {tuple(logits.shape)}")
    assert logits.shape == (B, T - 1, VOCAB)

    # 2. generate dan caption_image untuk satu gambar
    words = model.generate(images[:1])
    sentence = model.caption_image(images[:1], DummyVocab())
    assert isinstance(words, list) and isinstance(sentence, str)
    assert model.training, "Mode model harus kembali ke train setelah generate"
    print(f"caption : '{sentence[:60]}...' (masih acak, belum dilatih)")

    # 3. Gradien sampai ke lapisan pertama encoder
    model.zero_grad()
    model(images, captions).sum().backward()
    assert model.encoder.features[0].block[0].weight.grad.abs().sum() > 0
    print("gradien : mengalir dari keluaran decoder sampai Conv pertama encoder")

    # 4. Uji hafalan: latih pada 4 pasangan acak
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    targets = captions[:, 1:]                                   # (B, 29)

    print(f"\nUji hafalan (loss awal ideal ~ ln({VOCAB}) = {math.log(VOCAB):.2f}):")
    start = time.time()
    model.train()
    for step in range(1, 151):
        logits = model(images, captions)                        # (B, 29, V)
        loss = criterion(logits.reshape(-1, VOCAB), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1:
            first_loss = loss.item()
        if step in (1, 25, 50, 100, 150):
            print(f"  langkah {step:3d}: loss {loss.item():.4f}")
    final_loss = loss.item()
    print(f"  waktu: {time.time() - start:.1f} detik")

    assert final_loss < first_loss * 0.1, "Loss tidak turun cukup: kemungkinan ada bug"

    # 5. Setelah hafal, generate (mode eval) harus menulis ulang caption yang dihafal.
    #    BatchNorm di mode eval memakai statistik yang terkumpul selama training;
    #    karena batch-nya selalu sama, statistik itu cocok dengan 4 gambar ini.
    hafal = 0
    for b in range(B):
        pred = model.generate(images[b : b + 1])
        truth = captions[b, 1:12].tolist()                      # kata asli tanpa <start>/<end>
        hafal += int(pred == truth)
    print(f"  caption yang ditulis ulang dengan benar: {hafal}/{B}")
    assert hafal >= 3, "Model belum bisa menulis ulang caption yang dihafal"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nJumlah parameter (vocab {VOCAB}): {n_params:,}")
    print(f"Device: {device}")
    print("LULUS: CaptioningModel")


if __name__ == "__main__":
    test_model()
