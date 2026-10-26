# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 26/10/2026
- **Anggota:** Mazmur Bryan - Anggota 1 (Ketua), sebagai backup tahap optimasi/tuning
- **Target Modul:** Review PR tahap 2-3, download dataset, tuning hyperparameter (exp02, exp03), demo caption foto (`src/predict.py`)
- **LLM yang dipakai:** Claude (Claude Code)
- **Tautan Sesi Chat:** [lampirkan ekspor transkrip sesi Claude Code]
- **Konteks untuk LLM:** README (kontrak dan aturan from scratch), kode `src/train.py`, `src/evaluate.py`, `src/metrics.py` dari Anggota 3, dan grafik training exp01

Alur kerja vibe coding pada sesi ini: ketua memberi arahan lewat prompt, LLM menganalisis lalu menulis kode/notebook, ketua menjalankannya (Colab/laptop) dan melaporkan hasil atau error, lalu LLM memperbaiki atau melanjutkan.

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI

#### 1a. Mengecek progres tahap 2-3 di GitHub
- **Prompt Kelompok:**
  > "saya ingin lanjutkan ke tahap 2 dan 3, tapi jangan local saja atau google colab ... jelaskan simple dulu atau buat prompt atau check apakah data untuk bagian 2 sudah ada di github"
- **Respons LLM:** meninjau 2 PR yang masuk:
  - **PR #4 (data pipeline):** dinilai sesuai kontrak README (shape, token khusus, split seed 42, vocabulary dari data train saja, tanpa horizontal flip). Rekomendasi: merge.
  - **PR #5 (preprocess alternatif):** dinilai tidak layak, karena menghapus stopword ("a", "the", "is") sehingga caption menjadi kalimat patah, dan cara split-nya berbeda dari kontrak.
- **Tindak lanjut ketua:** me-merge PR #4. Setelah merge, LLM menjalankan uji integrasi data + model dengan dataset palsu kecil: shape `(4, 3, 128, 128)` → logits `(4, 29, V)` berhasil.

#### 1b. Membersihkan branch training (tahap 3)
- **Prompt Kelompok:**
  > "jelaskan simple masalah sekarang, intinya anda kerjakan saja dulu dan bersihkan"
- **Respons LLM:** menemukan branch `anggota3-training` memiliki riwayat Git yang terputus dari `main` (berasal dari salinan repo lain), sehingga tidak bisa di-PR normal, dan ikut mengubah notebook anggota lain. LLM membuat branch `anggota3-training-bersih` dari `main` dan memindahkan hanya commit tahap 3 (penulis commit asli tetap dipertahankan), tanpa perubahan notebook. Uji: 26 lulus.
- **Tindak lanjut ketua:** membuat dan me-merge PR #6.

#### 1c. Merancang tuning
- **Prompt Kelompok:**
  > "saya yang bakal backup jadi bagaimana, simple kelanjutannya"
- **Respons LLM:** mengusulkan 2 percobaan, masing-masing hanya mengubah **satu** pengaturan, dengan alasan yang diambil dari grafik exp01:

  | Percobaan | Yang diubah | Alasan dari grafik exp01 |
  |---|---|---|
  | exp01 (baseline, Anggota 3) | lr 1e-3, dropout 0.3 | Pembanding |
  | exp02 | learning rate **5e-4** | Val loss baru turun lagi setelah scheduler menurunkan LR di epoch 27 → LR awal mungkin terlalu besar |
  | exp03 | dropout **0.5** | Jarak train-val melebar setelah epoch ~7 → mulai overfitting |

  Pengaturan lain sama dengan exp01: seed 42, split 6091/1000/1000, batch 64, maks 30 epoch, early stopping patience 5.
- **Prompt Kelompok:**
  > "nomor 1 sudah dan anda check, nomor 2 mereka belum upload, jadi kita yang harus download"
  >
  > "berikan linknya colabnya, untuk saya run"
- **Respons LLM:** membuat `notebooks/download_dataset.ipynb` (unduh Flickr8k dari Kaggle langsung ke Google Drive, lalu disimpan sebagai `archive.zip`).
- **Hasil dijalankan ketua di Colab:**
  ```
  Selesai: 1113 MB -> /content/drive/MyDrive/ImageCaptioning/data/archive.zip
  ```
- **Respons LLM berikutnya:** membuat `notebooks/anggota1_tuning.ipynb` untuk menjalankan exp02 dan exp03, **otomatis melanjutkan dari checkpoint terakhir** jika Colab terputus, evaluasi, lalu tabel dan grafik perbandingan. Sebelum diberikan, perintah train/resume/evaluate diuji LLM di laptop dengan dataset palsu kecil.

#### 1d. Memindahkan training ke laptop
- **Prompt Kelompok:**
  > "bagaimana kalau kita tuning di local di laptop supaya jaga-jaga jika colab terputus"
- **Respons LLM:** mengecek laptop dan menemukan GPU **NVIDIA GTX 1660 Ti (6 GB)** serta RAM 7,8 GB. Usulnya: Colab menjalankan exp02 dan laptop exp03, supaya selesai 2x lebih cepat. Untuk itu PyTorch versi CUDA perlu dipasang dan dataset perlu diunduh ke laptop.
- **Prompt Kelompok:**
  > "ya, install"
- **Hasil:** PyTorch `2.14.1+cu126` terpasang, `cuda True | NVIDIA GeForce GTX 1660 Ti`. exp03 mulai berjalan di laptop (±85 detik per epoch).
- **Prompt Kelompok:**
  > "exp02 belum selesai di colab, apakah anda bisa jalankan disini? mungkin colabnya yang lambat"
- **Respons LLM:** menjalankan exp02 di laptop juga (±2,5 menit per epoch). Hasil dari Colab akhirnya tidak dipakai.

### 2. TAHAP PENANGANAN EROR / DEBUGGING

#### Error 1: PyTorch tidak memakai GPU
- **Pesan:** `torch 2.14.1+cpu | cuda False`, lalu `AssertionError: Torch not compiled with CUDA enabled`
- **Analisis & Solusi:** PyTorch yang terpasang adalah versi CPU. Pemasangan pertama versi CUDA tidak mengganti apa pun, karena pip menganggap `torch==2.14.1` sudah ada. Solusinya memasang ulang dengan `--force-reinstall` untuk versi `+cu126` (sesuai driver NVIDIA yang mendukung CUDA 12.5).

#### Error 2: training exp03 berhenti di epoch 24 (RAM penuh)
- **Pesan Eror:**
  ```
  RuntimeError: Caught RuntimeError in DataLoader worker process 2.
  RuntimeError: Couldn't open shared file mapping: <torch_27580_1146490213_5486>, error code: <1455>
  ```
- **Analisis & Solusi:**
  - Penyebab: kode error Windows 1455 berarti *paging file is too small*. Training memakai 4 proses pembaca data (`--num-workers 4`), masing-masing memuat salinan PyTorch ke memori, sementara RAM laptop hanya 7,8 GB (sisa ±1,5 GB saat error).
  - Solusi: **melanjutkan dari checkpoint epoch 23** (`--resume last_checkpoint.pt`) dengan `--num-workers 1`. Tidak ada epoch yang hilang karena checkpoint disimpan setiap epoch.
  - Pelajaran: di laptop dengan RAM terbatas, jumlah proses pembaca data harus kecil. Checkpoint per epoch terbukti menyelamatkan training.

#### Kendala 3: training di Colab lambat
- **Analisis:** setiap epoch membaca dan me-resize ±30.000 pasangan gambar-caption. Di Colab gratis pekerjaan ini dilakukan CPU 2 core, sehingga GPU banyak menunggu data. Solusinya, kedua percobaan dipindah ke laptop.

### 3. TAHAP OPTIMASI - Hasil tuning

| Percobaan | Epoch dijalankan | Epoch terbaik | Val loss terbaik | Test loss | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | CIDEr |
|---|---|---|---|---|---|---|---|---|---|---|
| exp01 (lr 1e-3, dropout 0.3) | 30 | 28 | **2,8945** | **2,9035** | 0,4605 | 0,2737 | 0,1539 | 0,0923 | **0,2962** | 0,3093 |
| exp02 (lr 5e-4) | 20 (early stop) | 15 | 2,9844 | 3,0091 | 0,4869 | **0,2926** | **0,1709** | **0,1048** | 0,2938 | **0,3593** |
| exp03 (dropout 0.5) | 25 (early stop) | 20 | 2,9468 | 2,9535 | **0,4871** | 0,2869 | 0,1642 | 0,0994 | 0,2826 | 0,3287 |

Grafik: `results/tuning_loss.png`, tabel: `results/tuning_perbandingan.csv`.

**Analisis (didiskusikan bersama LLM):**
1. **exp02 (LR lebih kecil) justru paling cepat overfitting.** Train loss turun paling jauh (2,58), tetapi val loss naik-turun dan berhenti di epoch 20. Hipotesis "LR terlalu besar" tidak terbukti untuk val loss.
2. **exp03 (dropout 0,5) paling stabil.** Jarak train-val paling kecil, jadi dropout yang lebih besar berhasil menahan overfitting.
3. **Loss dan metrik caption tidak sejalan.** exp01 punya loss terendah, tetapi exp02 dan exp03 punya BLEU dan CIDEr lebih tinggi. Cross-entropy mengukur keyakinan model pada setiap kata, sedangkan BLEU/CIDEr mengukur kemiripan kalimat utuh dengan caption manusia.

**Keputusan model final:** sesuai protokol README, model dipilih berdasarkan **validation loss** (bukan test score, agar tidak terjadi kebocoran data uji). Maka **exp01 tetap menjadi model final**. exp02 dicatat sebagai pengaturan dengan kualitas caption tertinggi pada test set.

**Keterbatasan:**
- exp01 dilatih di komputer Anggota 3, sedangkan exp02 dan exp03 di laptop ketua (GPU berbeda), sehingga ada sedikit perbedaan acak.
- Kriteria pemilihan yang lebih tepat adalah metrik caption (BLEU) pada validation set. Ini dapat menjadi pengembangan berikutnya.

### 4. UJI COBA PADA FOTO

- **Prompt Kelompok:**
  > "kalau kita mau coba input foto bagaimana atau lihat hasilnya"
- **Respons LLM:** membuat `src/predict.py`. Foto (file atau folder) masuk, caption keluar. Vocabulary dibaca langsung dari checkpoint, sehingga dataset tidak dibutuhkan. Opsi `--save` menyimpan grid gambar beserta caption.
  ```bash
  python -m src.predict --checkpoint best_model.pt --image folder_foto/ --save hasil.png
  ```
- **Hasil uji** pada 6 gambar test yang tidak pernah dilihat model (checkpoint exp02), hasilnya di `results/demo_caption.png`:

  | Caption model | Caption asli (manusia) | Penilaian |
  |---|---|---|
  | a dog is running through the grass | A brown and white dog is jumping up, with a stick in its mouth | Objek benar |
  | a man in a black shirt is climbing a rock | Three boys walk on rocks along a river | Objek dan tempat benar |
  | a man in a red shirt and a white shirt is standing on a sidewalk | A person is walking on a sidewalk ... | Tempat benar |
  | a man and woman are standing in front of a `<unk>` | A football player is running with the ball | Kurang tepat |
  | a man and a woman are sitting on a bench | Two children sit on an animal's back | Kurang tepat |
  | a woman in a red shirt and a white shirt is standing on a sidewalk | Kids playing in a tub of plastic balls | Salah |

- **Analisis:** model mengenali objek dan tempat yang sering muncul di Flickr8k (anjing, rumput, batu, trotoar), tetapi kesulitan dengan adegan yang jarang muncul (kolam bola, football). Penyebabnya: encoder CNN dilatih dari nol dengan hanya ±6.000 gambar, dan gambar diperkecil ke 128×128.

---

**File terkait:**
- `notebooks/download_dataset.ipynb`, `notebooks/anggota1_tuning.ipynb`
- `results/exp02_lr5e-4/`, `results/exp03_dropout0.5/` (riwayat training, evaluasi, grafik, log training)
- `results/tuning_perbandingan.csv`, `results/tuning_loss.png`
- `src/predict.py`, `results/demo_caption.png`
