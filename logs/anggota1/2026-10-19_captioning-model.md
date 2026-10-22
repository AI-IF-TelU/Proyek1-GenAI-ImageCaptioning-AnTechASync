# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 19/10/2026
- **Anggota:** Mazmur Bryan - Anggota 1 (Ketua, Prompt Engineer & Model)
- **Target Modul:** `CaptioningModel` (`src/model.py`), integrasi encoder + decoder
- **LLM yang dipakai:** Claude (Claude Code)
- **Tautan Sesi Chat:** [lampirkan ekspor transkrip sesi Claude Code]
- **Konteks untuk LLM:** blueprint, kontrak antarmodul, dan aturan from scratch di `README.md`

---

## KEPUTUSAN & ARAHAN KETUA

- **Integrasi sebagai modul terakhir Anggota 1.** Ketua memutuskan menggabungkan encoder dan decoder menjadi satu model utuh sebelum diserahkan ke Anggota 3 (training).
- **Verifikasi:** ketua menjalankan uji hafalan di GPU Colab dan memeriksa hasilnya (loss turun, 4/4 caption tepat).
- **Peninjauan log:** ketua memeriksa isi log sesi ini:
  > "dalam log itu coba tampilkan isinya apa?"
- **Integrasi ke `main`:** sebagai ketua, Mazmur Bryan membuat Pull Request #1 (`anggota1-model` -> `main`), merapikan judulnya, lalu melakukan merge. Setelah itu branch anggota 2-4 diperbarui agar berisi kode model terbaru.

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI
- **Prompt Kelompok:**
  > "sekarang lanjutkan ke CaptioningModel"

  Spesifikasi yang dirujuk (blueprint `README.md`):
  `CaptioningModel` menggabungkan `EncoderCNN` dan `DecoderLSTM`, dengan `forward(gambar, caption) -> logits (B, 29, vocab_size)`.
- **Alasan Khusus:**
  - Modul terakhir jobdesk Anggota 1.
  - Model utuh ini yang nanti dipanggil oleh `train.py` (Anggota 3). Sebelum diserahkan, harus dipastikan semua bagian benar-benar tersambung dan model mampu belajar, bukan hanya shape-nya yang cocok.
- **Output Model:**
  - `src/model.py`: class `CaptioningModel(vocab_size, feature_dim=256, embed_dim=256, hidden_dim=512, dropout=0.3)`
    - `forward(images, captions)`: encoder menghasilkan fitur, lalu decoder menghasilkan logits. Encoder dan decoder dilatih bersama: gradien loss mengalir lewat decoder ke encoder.
    - `generate(image)`: caption satu gambar dalam bentuk nomor kata. Otomatis berpindah ke mode `eval` (Dropout mati, BatchNorm memakai statistik training), lalu kembali ke mode semula.
    - `caption_image(image, vocab)`: sama dengan `generate`, tetapi hasilnya kalimat lewat `vocab.decode()` milik Anggota 2.
  - `tests/test_model.py`:
    1. Shape ujung ke ujung
    2. `generate` / `caption_image` berjalan dan mode model kembali ke `train`
    3. Gradien mengalir dari keluaran decoder sampai Conv pertama encoder
    4. **Uji hafalan (overfit test)**: model dilatih 150 langkah pada 4 pasangan gambar-caption acak (Adam lr 1e-3, `CrossEntropyLoss(ignore_index=0)`, gradient clipping 5.0). Loss harus turun di bawah 10% dari loss awal.
    5. Setelah hafal, `generate` harus menulis ulang caption yang dihafal dengan tepat.
  - Vocabulary asli (Anggota 2) belum ada, jadi uji memakai `DummyVocab` sebagai pengganti sementara.
  - Alasan uji hafalan yang dijelaskan LLM: jika seluruh aliran data dan gradien benar, model pasti bisa menghafal data sekecil ini. Jika tidak bisa, berarti ada bug. Uji ini menemukan masalah lebih cepat dan murah daripada langsung training penuh di Flickr8k.
  - Hasil uji yang dijalankan ketua di Colab (GPU Tesla T4):
    ```
    forward : gambar (4, 3, 128, 128) + caption (4, 30) -> logits (4, 29, 1000)
    caption : 'w156 w789 w547 w772 w599 w253 w253 w751 w288 w288 w825 w82 w...' (masih acak, belum dilatih)
    gradien : mengalir dari keluaran decoder sampai Conv pertama encoder

    Uji hafalan (loss awal ideal ~ ln(1000) = 6.91):
      langkah   1: loss 6.9234
      langkah  25: loss 0.4399
      langkah  50: loss 0.0454
      langkah 100: loss 0.0122
      langkah 150: loss 0.0066
      waktu: 4.5 detik
      caption yang ditulis ulang dengan benar: 4/4

    Jumlah parameter (vocab 1000): 3,652,104
    Device: cuda
    LULUS: CaptioningModel
    ```
  - Analisis hasil:
    - Loss awal 6.92, hampir sama dengan `ln(1000) = 6.91`, yaitu loss tebakan acak di antara 1000 kata. Artinya model mulai dari kondisi netral, sesuai teori.
    - Loss turun ke 0.0066 dan 4/4 caption ditulis ulang dengan tepat. Model mampu belajar dari ujung ke ujung.
    - Di GPU T4 uji hafalan butuh 4.5 detik, dibanding 40.4 detik di laptop (CPU), sekitar 9x lebih cepat.
    - Jumlah parameter `3,652,104` = encoder `1,045,024` + decoder `2,607,080`. Dengan vocabulary asli (~3.000 kata) jumlahnya akan lebih besar karena embedding dan lapisan keluaran ikut membesar.

### 2. TAHAP PENANGANAN EROR / DEBUGGING
Tidak ada error saat dijalankan.

Satu perbaikan desain sebelum uji dijalankan: versi awal uji hafalan sempat menyalakan mode `train` pada encoder saat `generate` agar BatchNorm memakai statistik batch. Cara itu juga menyalakan Dropout, sehingga hasilnya acak dan tidak valid. Diganti dengan `model.generate()` biasa (mode `eval`). Karena training selalu memakai batch yang sama, statistik BatchNorm yang terkumpul sudah cocok dengan 4 gambar tersebut.

### 3. TAHAP OPTIMASI
Belum ada. Optimasi dilakukan setelah training penuh dengan data Flickr8k (bersama Anggota 3).

---

**Commit terkait:**
- `abe8f2c` [model] tambah CaptioningModel (encoder + decoder) dan uji hafalan
- Pull Request #1 `anggota1-model` -> `main`, di-merge oleh ketua
