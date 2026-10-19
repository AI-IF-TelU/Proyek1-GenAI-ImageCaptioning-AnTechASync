# Image Captioning from Scratch

Proyek 1 CAK4KBB3: Vibe Coding Generative AI from Scratch.

Model menerima gambar dan menghasilkan satu kalimat deskripsi berbahasa Inggris.
Encoder Vanilla CNN dan decoder LSTM (sel ditulis manual) dibangun dan dilatih dari nol dengan PyTorch di Google Colab, memakai dataset Flickr8k.

## Kelompok

**Nama kelompok:** _[isi]_

| Anggota | Nama | Peran | Branch |
|---|---|---|---|
| 1 | Mazmur Bryan | Ketua, Prompt Engineer & Model | `anggota1-model` |
| 2 | _[isi]_ | Data Pipeline & Preprocessing | `anggota2-data` |
| 3 | _[isi]_ | Training Loop & Hyperparameter Tuning | `anggota3-training` |
| 4 | _[isi]_ | Dokumentasi & Error Handler | `anggota4-docs` |

## Aturan from scratch

- **Dilarang** model pretrained (ResNet, VGG, dll.).
- **Dilarang** `nn.LSTM`, `nn.LSTMCell`, `nn.RNN`, `nn.GRU`, `nn.Transformer*`, `nn.MultiheadAttention`.
- **Boleh** lapisan dasar: `nn.Linear`, `nn.Conv2d`, `nn.BatchNorm2d`, `nn.Embedding`, `nn.Dropout`, `nn.AdaptiveAvgPool2d`, fungsi aktivasi, loss, optimizer, `torchvision.transforms`.

## Kontrak antarmodul

Nama class, parameter, dan shape di bawah **tidak boleh diubah** tanpa persetujuan kelompok.

```
Gambar (B,3,128,128) → EncoderCNN → fitur (B,256) → DecoderLSTM → logits (B,29,V)
                                                         ↑
                                         caption (B,30)  [teacher forcing]
```

| Bagian | Masuk | Keluar |
|---|---|---|
| Data (Anggota 2) | file gambar + caption | gambar `(B, 3, 128, 128)` float32 dinormalisasi ke [-1, 1], caption `(B, 30)` int64 |
| `EncoderCNN` (Ketua) | gambar `(B, 3, 128, 128)` | fitur `(B, 256)` |
| `DecoderLSTM` (Ketua) | fitur `(B, 256)` + caption `(B, 30)` | logits `(B, 29, vocab_size)` |
| Training (Anggota 3) | logits + caption | loss |

- **Token khusus:** `<pad>`=0, `<start>`=1, `<end>`=2, `<unk>`=3
- **Caption:** maksimal 30 token termasuk `<start>` dan `<end>`. Kata yang muncul < 5 kali di data train menjadi `<unk>`.
- **Teacher forcing:** input decoder `caption[:, :-1]`, target `caption[:, 1:]`
- **Pembagian data:** per gambar, seed 42. 1000 val, 1000 test, sisanya train.

### Ukuran

| Nama | Nilai | Boleh di-tuning? |
|---|---|---|
| `IMG_SIZE` | 128 | Tidak |
| `FEATURE_DIM` | 256 | Tidak |
| `MAX_LEN` | 30 | Tidak |
| `EMBED_DIM` | 256 | Ya |
| `HIDDEN_DIM` | 512 | Ya |
| `DROPOUT` | 0.3 | Ya |
| `BATCH_SIZE` | 64 | Ya |
| `LEARNING_RATE` | 1e-3 | Ya |
| `WEIGHT_DECAY` | 1e-4 | Ya |
| `GRAD_CLIP` | 5.0 | Ya |

## Blueprint model

### 1. EncoderCNN (Vanilla CNN)
5 blok, tiap blok `Conv2d(3×3, padding=1, bias=False) → BatchNorm2d → ReLU → MaxPool2d(2)`

| Blok | Channel | Ukuran keluaran |
|---|---|---|
| input | 3 | 128×128 |
| 1 | 3 → 32 | 64×64 |
| 2 | 32 → 64 | 32×32 |
| 3 | 64 → 128 | 16×16 |
| 4 | 128 → 256 | 8×8 |
| 5 | 256 → 256 | 4×4 |

Lalu Global Average Pooling → `Linear(256, 256)` → `Dropout` → fitur `(B, 256)`.

### 2. LSTMCell manual
```
gates = Linear(input_size + hidden_size → 4 × hidden_size)([x, h])
i, f, g, o = pisah 4 bagian
i, f, o = sigmoid ; g = tanh
c' = f·c + i·g
h' = o·tanh(c')
```
Bias forget gate diinisialisasi 1.

### 3. DecoderLSTM
1. `Embedding(vocab_size, 256)`
2. Keadaan awal dari fitur gambar: `h0 = tanh(Linear(fitur))`, `c0 = tanh(Linear(fitur))`
3. Loop tiap langkah waktu memakai `LSTMCell` manual (hidden 512)
4. `Dropout(0.3)` → `Linear(512, vocab_size)` → logits
5. `generate(fitur, max_len=30)`: greedy search, mulai dari `<start>`, berhenti di `<end>`

### 4. CaptioningModel
Menggabungkan `EncoderCNN` dan `DecoderLSTM`. `forward(gambar, caption) → logits (B, 29, vocab_size)`.

### Training (acuan Anggota 3)
- Loss `CrossEntropyLoss(ignore_index=0)`, Adam (LR 1e-3, weight decay 1e-4), gradient clipping 5.0
- `ReduceLROnPlateau` dan early stopping berdasarkan val loss
- Augmentasi gambar **tanpa horizontal flip** (caption bisa memuat "left/right")
- Uji dulu dengan ±100 gambar sampai model bisa "menghafal", baru training penuh

## Struktur folder

```
image-captioning/
├── src/
│   ├── encoder.py      Ketua
│   ├── lstm_cell.py    Ketua
│   ├── decoder.py      Ketua
│   ├── model.py        Ketua
│   ├── vocab.py        Anggota 2
│   ├── dataset.py      Anggota 2
│   ├── train.py        Anggota 3
│   └── evaluate.py     Anggota 3
├── notebooks/          colab_runner.ipynb
├── tests/              uji tiap modul
├── logs/               log interaksi LLM per anggota
├── results/            grafik dan contoh caption
└── laporan/            laporan akhir
```

Dataset dan checkpoint disimpan di Google Drive (`/content/drive/MyDrive/ImageCaptioning`), bukan di repo.

## System prompt bersama (Bagian Umum)

Tempel di awal **setiap** chat baru dengan LLM, lalu tambahkan bagian peran masing-masing.

```
Kamu membantu kelompok kami membangun model Image Captioning dari nol (from scratch)
untuk tugas kuliah Generative AI dengan metode Vibe Coding. Model menerima gambar dan
menghasilkan satu kalimat deskripsi berbahasa Inggris.

SPESIFIKASI
- Framework: PyTorch, dijalankan di Google Colab (GPU T4).
- Dataset: Flickr8k (8.091 gambar, 5 caption per gambar).
- Arsitektur: Encoder Vanilla CNN -> vektor fitur -> Decoder LSTM manual -> kalimat.

ATURAN WAJIB
1. From scratch: DILARANG model pretrained (ResNet, VGG, dll.) dan DILARANG nn.LSTM,
   nn.LSTMCell, nn.RNN, nn.GRU, nn.Transformer*, nn.MultiheadAttention.
   Yang boleh: lapisan dasar seperti nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.Embedding,
   nn.Dropout, nn.AdaptiveAvgPool2d, fungsi aktivasi, loss, optimizer, torchvision.transforms.
2. Jangan berasumsi. Jika ada yang ambigu, tanyakan dulu sebelum menulis kode.
3. Ikuti kontrak di bawah. Nama class, parameter, dan shape tidak boleh diubah.
4. Kode modular, rapi, diberi komentar singkat Bahasa Indonesia, dan setiap langkah
   penting diberi komentar shape tensor, misalnya  # (B, 256)
5. Jelaskan alasan setiap keputusan desain secara singkat, karena kami harus bisa
   menjelaskannya saat presentasi.
6. Sertakan kode uji kecil memakai tensor acak (torch.randn) untuk memastikan shape benar.
7. Jika ada error, jelaskan penyebabnya dulu, baru beri perbaikan.

KONTRAK
- Gambar:   (B, 3, 128, 128), float32, dinormalisasi ke [-1, 1]
- Caption:  (B, 30), int64. Token khusus: <pad>=0, <start>=1, <end>=2, <unk>=3
- EncoderCNN:  (B, 3, 128, 128) -> fitur (B, 256)
- DecoderLSTM: fitur (B, 256) + caption (B, 30) -> logits (B, 29, vocab_size)
  Teacher forcing: input = caption[:, :-1], target = caption[:, 1:]
- Ukuran: FEATURE_DIM=256, EMBED_DIM=256, HIDDEN_DIM=512, DROPOUT=0.3, MAX_LEN=30
```

## Alur kerja

1. `git pull origin main` sebelum mulai bekerja.
2. Kerjakan modul di branch masing-masing, bersama LLM memakai system prompt di atas.
3. Uji di Colab lewat `notebooks/colab_runner.ipynb`.
4. Commit sedikit-sedikit dengan pesan jelas, misalnya `[model] tambah EncoderCNN`.
5. Isi log di `logs/anggotaX/` memakai [`logs/TEMPLATE_LOG.md`](logs/TEMPLATE_LOG.md) pada hari yang sama.
6. Modul selesai: buat Pull Request ke `main`. Ketua meninjau lalu merge.
