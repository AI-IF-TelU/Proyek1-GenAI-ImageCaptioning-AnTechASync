# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 07/10/2026 - 19/10/2026 (perencanaan), 19/10/2026 (setup dan EncoderCNN)
- **Anggota:** Mazmur Bryan - Anggota 1 (Ketua, Prompt Engineer & Model)
- **Target Modul:** Perencanaan arsitektur, setup repo dan notebook Colab, `EncoderCNN` (`src/encoder.py`)
- **LLM yang dipakai:** Claude (Claude Code)
- **Tautan Sesi Chat:** [lampirkan ekspor transkrip sesi Claude Code]
- **Konteks untuk LLM:** blueprint, kontrak antarmodul, dan aturan from scratch di `README.md`

---

## PEMBAGIAN PERAN DALAM SESI

| Ketua (Mazmur Bryan) | LLM (Claude) |
|---|---|
| Menentukan model, aturan from scratch, dan batasan teknis | Menjelaskan konsep dan memberi pilihan beserta konsekuensinya |
| Meminta peninjauan blueprint lalu menyetujuinya | Meninjau blueprint dan mengusulkan perbaikan |
| Menyiapkan repo GitHub, folder Google Drive, dan runtime Colab | Menulis struktur repo, README, notebook, dan kode sesuai arahan |
| Menjalankan semua uji di GPU Colab dan melaporkan hasil/error | Menganalisis error dan menulis perbaikannya |

---

## KEPUTUSAN & ARAHAN KETUA (sesi perencanaan)

Sebelum menulis kode, arsitektur ditentukan lewat beberapa sesi diskusi. Keputusan penting yang diambil ketua:

1. **From scratch secara ketat.** LLM sempat menyarankan encoder ResNet pretrained. Ketua menolak:
   > "harus yang diminta, jika dari scratch ya dari scratch, jangan buat asumsi"

   Akibatnya, encoder CNN dan decoder dibangun dari bobot acak tanpa model pretrained.
2. **Klarifikasi peran CNN.** Ketua mempertanyakan kenapa ada CNN jika harus dari nol:
   > "maksudnya anda katakan semua dari nol tapi kenapa pakai CNN, atau kita tidak pakai CNN?"

   Kesimpulan: konsep CNN tetap dipakai sesuai dokumen tugas, tetapi lapisannya disusun dan dilatih sendiri.
3. **Decoder LSTM ditulis manual.** Ketua memilih LSTM, lalu memutuskan tidak memakai `nn.LSTM` bawaan:
   > "untuk sekarang manual, karena kalau yang sudah bawaan bisa dicela"
4. **Encoder vanilla CNN.** Ketua menanyakan kesesuaiannya dengan syarat tugas:
   > "kalau pakai vanilla CNN sesuai dengan requirement atau tidak?"

   Vanilla CNN dipilih karena sesuai syarat dan mudah dijelaskan saat presentasi.
5. **Peninjauan blueprint.** Ketua meminta blueprint dicek sebelum dipakai:
   > "check apakah sudah baik blueprint ini, jika sudah baik, lanjutkan ke system prompt bersama"

   Hasilnya, 5 perbaikan diterima:
   - `Conv2d(bias=False)` sebelum BatchNorm
   - weight decay 1e-4
   - LR scheduler dan early stopping
   - tanpa horizontal flip, karena caption bisa memuat "left/right"
   - inisialisasi bobot khusus dijadikan opsional
6. **Platform dan alur kerja:** PyTorch, dataset Flickr8k, Google Colab (GPU T4). Kode disimpan di GitHub dan dijalankan di Colab, dataset dan checkpoint disimpan di Google Drive. Ketua memutuskan kode di-push bertahap, bukan sekaligus di akhir.
7. **Repo dan akun.** Ketua memutuskan memakai satu akun GitHub (`steezym`) dan repo public tanpa token:
   > "token github tidak diperlukan"

   Repo sempat dibuat ulang agar seluruh riwayat commit konsisten dengan satu akun.

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI

#### 1a. Setup repo dan notebook Colab
- **Prompt Kelompok:**
  > "maksudnya yang semua anda bisa buat, anda lakukan karena ini vibe coding, jika ada yang perlu saya buat manual contoh google colab atau apapun itu anda beri tahu, simple saja"
- **Alasan Khusus:** Ketua menetapkan pola kerja vibe coding: LLM menulis bagian yang bisa dikerjakan lewat kode, sedangkan ketua mengerjakan langkah manual (Colab, Drive, pengaturan GitHub) dan memverifikasi hasilnya. Sebelum menulis model, semua anggota butuh tempat kerja yang sama (struktur folder, aturan from scratch, kontrak shape, notebook Colab) agar bisa bekerja paralel.
- **Output Model:**
  - Struktur folder `src/`, `notebooks/`, `tests/`, `logs/anggota1-4/`, `results/`, `laporan/`
  - `.gitignore` agar dataset dan checkpoint (`*.zip`, `*.pt`) tidak masuk repo
  - `README.md` berisi aturan from scratch, kontrak antarmodul, blueprint, dan system prompt bersama
  - `notebooks/colab_runner.ipynb`: mount Google Drive, clone/pull repo, pindah branch, cek GPU
  - Branch `anggota1-model` sampai `anggota4-docs`
- **Langkah manual ketua:** membuat repo GitHub, membuat folder Google Drive `ImageCaptioning`, membuka notebook di Colab dengan runtime T4, lalu memverifikasi `Drive OK: True` dan `GPU tersedia: True`.

#### 1b. EncoderCNN
- **Prompt Kelompok:**
  > "okay sekarang kerjakan encoderCNN"

  Spesifikasi yang dirujuk (blueprint yang sudah disetujui ketua di `README.md`):
  - 5 blok `Conv2d(3x3, padding=1, bias=False) -> BatchNorm2d -> ReLU -> MaxPool2d(2)`
  - channel 3 -> 32 -> 64 -> 128 -> 256 -> 256
  - Global Average Pooling -> `Linear(256, 256)` -> `Dropout`
  - keluaran `(B, 256)`
- **Alasan Khusus:**
  - Encoder dikerjakan pertama karena keluarannya `(B, 256)` menjadi masukan decoder.
  - Vanilla CNN sesuai keputusan ketua (from scratch, mudah dijelaskan).
  - Ukuran 128x128 dipilih agar ringan dilatih dari nol di GPU T4 gratis.
- **Output Model:**
  - `src/encoder.py`
    - `ConvBlock`: satu blok Conv -> BatchNorm -> ReLU -> MaxPool, dipakai ulang 5 kali.
    - `EncoderCNN`: 5 blok -> `AdaptiveAvgPool2d(1)` -> `Linear(256, 256)` -> `Dropout(0.3)`.
  - `tests/test_encoder.py`: uji dengan `torch.randn(4, 3, 128, 128)` untuk mengecek shape tiap blok, keluaran `(4, 256)`, aliran gradien sampai lapisan pertama, dan jumlah parameter.
  - Keputusan desain yang dijelaskan LLM:
    - `bias=False` pada Conv karena BatchNorm sudah punya parameter geser (beta), sehingga bias Conv mubazir.
    - Global Average Pooling menghasilkan 256 angka, bukan 4096 jika di-flatten (256x4x4). Linear sesudahnya jauh lebih kecil dan model tidak mudah overfit.
  - Hasil uji yang dijalankan ketua di Colab (GPU Tesla T4):
    ```
    input   : (4, 3, 128, 128)
    blok 1  : (4, 32, 64, 64)
    blok 2  : (4, 64, 32, 32)
    blok 3  : (4, 128, 16, 16)
    blok 4  : (4, 256, 8, 8)
    blok 5  : (4, 256, 4, 4)
    keluaran: (4, 256)
    Jumlah parameter: 1,045,024
    Device: cuda
    LULUS: EncoderCNN (4, 3, 128, 128) -> (4, 256)
    ```

### 2. TAHAP PENANGANAN EROR / DEBUGGING

#### Error 1: `autoreload` gagal di Colab
- **Pesan Eror dari Notebook** (ditemukan ketua saat menjalankan sel Setup):
  ```
  /usr/local/lib/python3.13/dist-packages/IPython/extensions/autoreload.py in <module>
  --> 121 from imp import reload

  ModuleNotFoundError: No module named 'imp'
  ```
- **Prompt Perbaikan ke LLM:** ketua mengirim traceback lengkap dari sel Setup Colab.
- **Analisis & Solusi Kelompok:**
  - Penyebab: runtime Colab memakai Python 3.13, sedangkan ekstensi `autoreload` dari IPython versi di Colab masih mengimpor modul `imp` yang sudah dihapus sejak Python 3.12. Ini masalah kompatibilitas lingkungan, bukan kesalahan kode kelompok.
  - Solusi: `%load_ext autoreload` diganti fungsi `muat_ulang()` yang menghapus modul `src` dari `sys.modules`, sehingga import berikutnya membaca kode terbaru setelah `git pull`.
  - Pelajaran: baca baris paling bawah traceback (`No module named 'imp'`) dan lokasi file-nya. Lokasi `dist-packages/IPython` menunjukkan sumber error ada di library, bukan di kode kita.

#### Error 2: modul uji tidak ditemukan di Colab
- **Pesan Eror dari Notebook** (ditemukan ketua saat menjalankan uji encoder):
  ```
  /usr/bin/python3: No module named tests.test_encoder
  ```
- **Prompt Perbaikan ke LLM:** ketua mengirim pesan error dari sel uji `!python -m tests.test_encoder`.
- **Analisis & Solusi Kelompok:**
  - Penyebab: Colab sudah punya paket lain bernama `tests` (bawaan salah satu library terpasang). Folder `tests/` kelompok belum punya `__init__.py`, jadi Python memprioritaskan paket `tests` milik Colab dan tidak menemukan `test_encoder` di sana. Di laptop error ini tidak muncul karena tidak ada paket `tests` lain.
  - Solusi: menambahkan file kosong `tests/__init__.py` agar folder `tests/` menjadi paket resmi yang didahulukan.
  - Pelajaran: kode yang lulus di laptop belum tentu lulus di lingkungan lain. Selalu uji di lingkungan target (Colab).

#### Error 3: push ditolak Git
- **Pesan Eror:**
  ```
  ! [rejected]  anggota1-model -> anggota1-model (fetch first)
  ```
- **Analisis & Solusi Kelompok:**
  - Penyebab: ketua menyimpan notebook ke GitHub langsung dari Colab (commit "Created using Colab"), sehingga branch di GitHub lebih baru daripada di laptop.
  - Solusi: `git pull --rebase` untuk menggabungkan perubahan dari Colab, lalu push ulang.
  - Pelajaran: selalu `git pull` sebelum push, terutama jika menyimpan dari dua tempat (laptop dan Colab).

### 3. TAHAP OPTIMASI
Belum ada. Encoder belum dilatih. Optimasi dilakukan setelah model lengkap dan training berjalan.

---

**Commit terkait** (branch `anggota1-model`):
- `9e6d2fa` [setup] struktur folder, README, template log, notebook Colab
- `a387fff` [setup] ganti autoreload dengan muat_ulang() karena gagal di Python 3.13
- `7704741` Created using Colab (notebook disimpan ketua dari Colab)
- `bd7690a` [model] tambah EncoderCNN (vanilla CNN 5 blok + GAP) dan uji shape
- `84ae917` [test] tambah tests/__init__.py agar tidak tertimpa paket 'tests' lain di Colab
