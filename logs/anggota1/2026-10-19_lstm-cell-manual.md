# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 19/10/2026
- **Anggota:** Mazmur Bryan - Anggota 1 (Ketua, Prompt Engineer & Model)
- **Target Modul:** `LSTMCell` manual (`src/lstm_cell.py`)
- **LLM yang dipakai:** Claude (Claude Code)
- **Tautan Sesi Chat:** [lampirkan ekspor transkrip sesi Claude Code]
- **Konteks untuk LLM:** blueprint, kontrak antarmodul, dan aturan from scratch di `README.md`

---

## KEPUTUSAN & ARAHAN KETUA

- **LSTM ditulis manual**, bukan `nn.LSTM` / `nn.LSTMCell`. Keputusan ini diambil ketua di sesi perencanaan agar implementasi tidak bisa dianggap memakai model jadi:
  > "untuk sekarang manual, karena kalau yang sudah bawaan bisa dicela"
- **Spesifikasi yang disetujui ketua di blueprint:**
  - satu `nn.Linear` untuk keempat gate
  - `i, f, o = sigmoid`, `g = tanh`
  - bias forget gate diinisialisasi 1
- **Urutan pengerjaan:** ketua menetapkan urutan modul Anggota 1, yaitu EncoderCNN -> LSTMCell -> DecoderLSTM -> CaptioningModel. LSTMCell dikerjakan sebelum decoder karena decoder memanggil sel ini di setiap langkah.
- **Verifikasi:** ketua menjalankan uji di GPU Colab dan memastikan semua pengecekan lulus sebelum lanjut ke modul berikutnya.

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI
- **Prompt Kelompok:**
  > "lanjut"

  Lanjutan dari langkah berikutnya yang sudah disepakati: LSTMCell manual sesuai blueprint, beserta penjelasan fungsi tiap gate.
- **Alasan Khusus:** sel LSTM adalah "memori" decoder. Menuliskannya manual memperlihatkan pemahaman tentang cara kerja gate, yang juga akan ditanyakan saat presentasi.
- **Output Model:**
  - `src/lstm_cell.py`: class `LSTMCell(input_size, hidden_size)`, `forward(x, (h, c)) -> (h', c')`.
    - Gabungan `[x, h]` masuk ke satu `nn.Linear`, hasilnya dipotong 4 menjadi `i, f, g, o`.
    - Fungsi tiap bagian:
      - `i` (input gate): seberapa banyak info baru dicatat
      - `f` (forget gate): seberapa banyak catatan lama disimpan
      - `g` (candidate): isi info baru
      - `o` (output gate): seberapa banyak catatan dikeluarkan
  - `tests/test_lstm_cell.py` mengecek 5 hal:
    1. Shape keluaran `(4, 512)`
    2. Rumus gate dihitung ulang secara terpisah dan hasilnya harus sama persis (membuktikan urutan `i, f, g, o` tidak tertukar)
    3. Bias forget gate = 1
    4. Nilai `h` di rentang -1..1
    5. Loop 30 langkah (sepanjang satu caption) dengan gradien tetap sampai ke kata pertama
  - Keputusan desain yang dijelaskan LLM:
    - **Satu Linear untuk 4 gate**: hasil matematisnya sama dengan 4 Linear terpisah, tetapi lebih cepat karena dihitung dalam satu perkalian matriks.
    - **Bias forget gate = 1**: di awal training `f ~ sigmoid(1) ~ 0.73`, sehingga sel cenderung menyimpan memori dan informasi awal kalimat tidak cepat hilang.
  - Hasil uji yang dijalankan ketua di Colab (GPU Tesla T4):
    ```
    Satu langkah : x (4, 256) -> h (4, 512), c (4, 512)
    Rumus gate   : sesuai (c' = f*c + i*g, h' = o*tanh(c'))
    Bias forget  : 1.0 (gate lain dibiarkan acak)
    Rentang h    : [-0.249, 0.305] (harus di dalam -1..1)
    30 langkah   : gradien ke kata pertama = 0.1518 (> 0)
    Jumlah parameter: 1,574,912
    Device: cuda
    LULUS: LSTMCell manual
    ```
    Di laptop (CPU) juga lulus. Angka rentang `h` dan gradien sedikit berbeda (`[-0.262, 0.255]`, `0.1540`) karena generator angka acak GPU berbeda dengan CPU, bukan karena bug.
  - Pengecekan jumlah parameter: `(256 + 512) x 2048 + 2048 = 1,574,912`, cocok dengan hasil uji.

### 2. TAHAP PENANGANAN EROR / DEBUGGING
Tidak ada error pada sesi ini. Pelajaran dari sesi sebelumnya sudah diterapkan: `tests/__init__.py` sudah ada, dan `git pull --rebase` dilakukan sebelum push.

### 3. TAHAP OPTIMASI
Belum ada. Sel akan dioptimasi bersama decoder setelah training berjalan.

---

**Commit terkait** (branch `anggota1-model`):
- `6e89ee8` [model] tambah LSTMCell manual (4 gate dari satu nn.Linear) dan uji
