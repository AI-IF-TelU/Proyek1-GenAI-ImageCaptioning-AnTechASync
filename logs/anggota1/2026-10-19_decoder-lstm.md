# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 19/10/2026
- **Anggota:** Mazmur Bryan - Anggota 1 (Ketua, Prompt Engineer & Model)
- **Target Modul:** `DecoderLSTM` (`src/decoder.py`)
- **LLM yang dipakai:** Claude (Claude Code)
- **Tautan Sesi Chat:** [lampirkan ekspor transkrip sesi Claude Code]
- **Konteks untuk LLM:** blueprint, kontrak antarmodul, dan aturan from scratch di `README.md`

---

## KEPUTUSAN & ARAHAN KETUA

- **Spesifikasi decoder yang disetujui ketua di blueprint:**
  - `Embedding(vocab_size, 256)`
  - memori awal dari fitur gambar `h0 = tanh(Linear(fitur))`, `c0 = tanh(Linear(fitur))`
  - `LSTMCell` manual (hidden 512)
  - `Dropout(0.3)` lalu `Linear(512, vocab_size)`
  - `generate` dengan greedy search
- **Kontrak keluaran** `(B, 29, vocab_size)` ditetapkan ketua sejak sesi perencanaan, agar `train.py` milik Anggota 3 bisa langsung memakainya tanpa koordinasi ulang.
- **Verifikasi:** ketua menjalankan uji di GPU Colab, memastikan semua pengecekan lulus, lalu memutuskan lanjut ke CaptioningModel.

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI
- **Prompt Kelompok:**
  > "lanjut"

  Lanjutan dari langkah berikutnya yang sudah disepakati: DecoderLSTM sesuai blueprint, beserta penjelasan teacher forcing dan cara fitur gambar masuk ke LSTM.
- **Alasan Khusus:** decoder adalah penghubung antara fitur gambar (keluaran encoder) dan kalimat. Dua konsep yang wajib bisa dijelaskan saat presentasi adalah teacher forcing dan cara gambar memengaruhi kalimat.
- **Output Model:**
  - `src/decoder.py`: class `DecoderLSTM(vocab_size, embed_dim=256, hidden_dim=512, feature_dim=256, dropout=0.3)`
    - `forward(features, captions)`: teacher forcing. Input decoder `captions[:, :-1]`, keluaran logits `(B, 29, vocab_size)`. Logits ke-t adalah tebakan untuk `captions[:, t+1]`.
    - `generate(features, max_len=30)`: greedy search untuk satu gambar. Mulai dari `<start>`, kata dengan skor tertinggi menjadi input langkah berikutnya, berhenti di `<end>`. Mengembalikan nomor kata tanpa `<start>`/`<end>`.
    - `nn.Embedding(..., padding_idx=0)`: vektor `<pad>` selalu nol dan tidak ikut dilatih.
  - `tests/test_decoder.py` mengecek 6 hal:
    1. Shape sesuai kontrak
    2. Teacher forcing tidak "mengintip" kata masa depan: kata ke-15 dst. diubah, tebakan langkah 0-14 harus tetap
    3. Fitur gambar benar-benar dipakai: gambar berbeda menghasilkan tebakan berbeda
    4. Embedding `<pad>` = 0
    5. Gradien mengalir sampai fitur gambar (agar encoder ikut belajar)
    6. `generate` menghasilkan daftar nomor kata yang valid
  - Keputusan desain yang dijelaskan LLM:
    - **Teacher forcing**: saat training, LSTM diberi kata yang benar dari caption asli, bukan tebakannya sendiri. Training lebih cepat dan stabil karena satu tebakan salah tidak merusak langkah berikutnya. Saat `generate` tidak ada caption asli, jadi model memakai tebakannya sendiri.
    - **Fitur gambar sebagai memori awal** (`h0`, `c0`): cara paling sederhana menyambungkan encoder dan decoder. `tanh` menjaga nilainya di rentang -1..1 seperti `h` biasa. Rencana optimasi jika caption melenceng dari gambar: memasukkan fitur gambar di setiap langkah.
  - Hasil uji yang dijalankan ketua di Colab (GPU Tesla T4):
    ```
    forward      : fitur (4, 256) + caption (4, 30) -> logits (4, 29, 1000)
    Teacher force: tebakan langkah t hanya bergantung pada kata 0..t (tidak mengintip)
    Fitur gambar : gambar berbeda menghasilkan tebakan berbeda
    Embedding pad: vektor <pad> = 0
    Gradien      : mengalir sampai fitur gambar
    generate     : 29 kata, contoh nomor kata [750, 568, 893, 893, 893, 913, 163, 913]...
                   (masih acak karena belum dilatih)
    Jumlah parameter (vocab 1000): 2,607,080
    Device: cuda
    LULUS: DecoderLSTM
    ```
  - Catatan: `generate` masih menghasilkan 29 kata acak tanpa `<end>`. Ini wajar karena model belum dilatih. Ukuran kamus 1000 hanya angka uji; nanti mengikuti vocabulary asli dari Anggota 2.
  - Pengecekan jumlah parameter:
    - embedding `1000 x 256 = 256,000`
    - `init_h` + `init_c` `2 x (256 x 512 + 512) = 263,168`
    - LSTMCell `1,574,912`
    - `fc_out` `512 x 1000 + 1000 = 513,000`
    - total `2,607,080`, cocok dengan hasil uji

### 2. TAHAP PENANGANAN EROR / DEBUGGING
Tidak ada error pada sesi ini.

### 3. TAHAP OPTIMASI
Belum ada. Kandidat optimasi setelah training: fitur gambar di setiap langkah, dan beam search menggantikan greedy.

---

**Commit terkait** (branch `anggota1-model`):
- `16b4be6` [model] tambah DecoderLSTM (teacher forcing + greedy generate) dan uji
