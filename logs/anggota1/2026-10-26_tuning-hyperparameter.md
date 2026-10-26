# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 26/10/2026
- **Anggota:** Mazmur Bryan - Anggota 1 (Ketua), sebagai backup tahap optimasi/tuning
- **Target Modul:** Tuning hyperparameter (exp02, exp03) dan perbandingan dengan exp01 milik Anggota 3
- **LLM yang dipakai:** Claude (Claude Code)
- **Tautan Sesi Chat:** [lampirkan ekspor transkrip sesi Claude Code]
- **Konteks:** `src/train.py`, `src/evaluate.py`, `src/metrics.py` (Anggota 3) dan hasil training exp01 di `results/`

---

## KEPUTUSAN & ARAHAN KETUA

- **Ketua mengambil alih tuning sebagai backup**, karena training awal (exp01) sudah dikerjakan Anggota 3 tetapi tuning wajib menurut dokumen tugas ("eksperimen fine-tuning: tuning learning rate, batch size"):
  > "saya yang bakal backup jadi bagaimana, simple kelanjutannya"
- **Dataset diunduh sendiri ke Google Drive**, karena belum diunggah anggota lain (`notebooks/download_dataset.ipynb`).
- **Training dipindah ke laptop sebagai cadangan Colab**, karena tenggat pengumpulan Rabu:
  > "bagaimana kalau kita tuning di local di laptop supaya jaga-jaga jika colab terputus"

  Laptop ternyata punya GPU NVIDIA GTX 1660 Ti, sehingga PyTorch versi CUDA dipasang dan exp03 dijalankan di laptop.
- **exp02 juga dipindah ke laptop** karena Colab berjalan lambat:
  > "exp02 belum selesai di colab, apakah anda bisa jalankan disini? mungkin colabnya yang lambat"

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI - Merancang percobaan

- **Prompt Kelompok:**
  > "saya yang bakal backup jadi bagaimana, simple kelanjutannya"
- **Alasan Khusus:** pengaturan yang diuji dipilih dari **pola grafik exp01**, bukan acak. Satu percobaan hanya mengubah **satu** pengaturan, agar pengaruhnya jelas.

  | Percobaan | Yang diubah | Alasan dari grafik exp01 |
  |---|---|---|
  | exp01 (baseline, Anggota 3) | lr 1e-3, dropout 0.3 | Pembanding |
  | exp02 | learning rate **5e-4** | Val loss exp01 baru turun lagi setelah scheduler menurunkan LR di epoch 27, jadi LR awal mungkin terlalu besar |
  | exp03 | dropout **0.5** | Jarak train-val exp01 melebar setelah epoch ~7 (mulai overfitting), jadi regularisasi lebih kuat mungkin membantu |

  Pengaturan lain sama dengan exp01: seed 42, split 6091/1000/1000, batch 64, max 30 epoch, early stopping patience 5, scheduler ReduceLROnPlateau (patience 2, factor 0.5).
- **Output Model:**
  - `notebooks/download_dataset.ipynb`: unduh Flickr8k dari Kaggle ke `MyDrive/ImageCaptioning/data/archive.zip` (1.113 MB, 8.091 gambar).
  - `notebooks/anggota1_tuning.ipynb`: menjalankan exp02 dan exp03 di Colab, **otomatis melanjutkan dari checkpoint terakhir** jika Colab terputus, evaluasi, lalu tabel dan grafik perbandingan.
  - Perintah train/resume/evaluate diuji dulu di laptop dengan dataset palsu kecil sebelum dijalankan pada data asli.

### 2. TAHAP PENANGANAN EROR / DEBUGGING

#### Error 1: training di laptop berhenti di epoch 24 (RAM penuh)
- **Pesan Eror:**
  ```
  RuntimeError: Caught RuntimeError in DataLoader worker process 2.
  RuntimeError: Couldn't open shared file mapping: <torch_27580_1146490213_5486>, error code: <1455>
  ```
- **Analisis & Solusi Kelompok:**
  - Penyebab: kode error Windows 1455 berarti *paging file is too small*. Training memakai 4 proses pembaca data (`--num-workers 4`), masing-masing memuat salinan PyTorch ke memori. Laptop hanya punya RAM 7,8 GB (sisa ~1,5 GB saat error).
  - Solusi: training **dilanjutkan dari checkpoint epoch 23** (`--resume last_checkpoint.pt`) dengan `--num-workers 1`. Tidak ada epoch yang hilang, karena checkpoint disimpan setiap epoch.
  - Pelajaran: di laptop dengan RAM terbatas, jumlah proses pembaca data harus kecil. Checkpoint per epoch terbukti penting.

#### Kendala 2: PyTorch di laptop tidak memakai GPU
- **Gejala:** `torch 2.14.1+cpu | cuda False`, padahal laptop punya GPU GTX 1660 Ti.
- **Solusi:** memasang ulang PyTorch versi CUDA 12.6 (`torch==2.14.1+cu126`, sesuai driver NVIDIA yang mendukung CUDA 12.5). Pemasangan pertama gagal karena pip menganggap versi 2.14.1 sudah terpasang, sehingga perlu `--force-reinstall`.

#### Kendala 3: training di Colab lambat
- **Analisis:** setiap epoch membaca dan me-resize ±30.000 pasangan gambar-caption. Di Colab gratis bagian ini dikerjakan CPU 2 core, sehingga GPU banyak menunggu data. exp02 akhirnya juga dijalankan di laptop (±2,5 menit per epoch).

### 3. TAHAP OPTIMASI - Hasil

| Percobaan | Epoch dijalankan | Epoch terbaik | Val loss terbaik | Test loss | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | CIDEr |
|---|---|---|---|---|---|---|---|---|---|---|
| exp01 (lr 1e-3, dropout 0.3) | 30 | 28 | **2,8945** | **2,9035** | 0,4605 | 0,2737 | 0,1539 | 0,0923 | **0,2962** | 0,3093 |
| exp02 (lr 5e-4) | 20 (early stop) | 15 | 2,9844 | 3,0091 | 0,4869 | **0,2926** | **0,1709** | **0,1048** | 0,2938 | **0,3593** |
| exp03 (dropout 0.5) | 25 (early stop) | 20 | 2,9468 | 2,9535 | **0,4871** | 0,2869 | 0,1642 | 0,0994 | 0,2826 | 0,3287 |

Grafik: `results/tuning_loss.png`, tabel: `results/tuning_perbandingan.csv`.

**Analisis:**
1. **exp02 (LR lebih kecil) justru paling cepat overfitting.** Train loss turun paling jauh (2,58), tetapi val loss naik-turun dan berhenti di epoch 20. Hipotesis awal "LR terlalu besar" tidak terbukti untuk val loss.
2. **exp03 (dropout 0,5) paling stabil.** Jarak train-val paling kecil di awal training. Dropout lebih besar berhasil menahan overfitting.
3. **Loss dan metrik caption tidak sejalan.** exp01 punya loss terendah, tetapi exp02 dan exp03 punya BLEU dan CIDEr lebih tinggi. Cross-entropy mengukur seberapa yakin model pada setiap kata, sedangkan BLEU/CIDEr mengukur kemiripan kalimat utuh hasil greedy decoding dengan caption manusia.

**Keputusan model final:** sesuai protokol di README, model dipilih berdasarkan **validation loss** (bukan test score, agar tidak terjadi kebocoran data uji). Maka **exp01 tetap menjadi model final**. exp02 dicatat sebagai pengaturan dengan kualitas caption (BLEU-4, CIDEr) tertinggi pada test set.

**Keterbatasan:**
- exp01 dilatih di komputer Anggota 3, sedangkan exp02 dan exp03 di laptop ketua (GPU berbeda), sehingga ada sedikit perbedaan acak non-deterministik.
- Pemilihan model memakai val loss. Metrik caption pada validation set (BLEU di val) akan menjadi kriteria yang lebih tepat. Ini dapat menjadi pengembangan berikutnya.

**Contoh caption exp02 (test set):**
- "a man in a wetsuit is surfing"
- "a brown dog is running on a grassy field"
- "a young boy in a blue shirt is swimming in a pool"
- Gagal: "two women are sitting on a `<unk>` `<unk>`" (kata objek tidak ada di kamus karena muncul < 5 kali di data train)

---

**File terkait:**
- `notebooks/download_dataset.ipynb`, `notebooks/anggota1_tuning.ipynb`
- `results/exp02_lr5e-4/`, `results/exp03_dropout0.5/` (riwayat training, evaluasi, grafik, log training)
- `results/tuning_perbandingan.csv`, `results/tuning_loss.png`
