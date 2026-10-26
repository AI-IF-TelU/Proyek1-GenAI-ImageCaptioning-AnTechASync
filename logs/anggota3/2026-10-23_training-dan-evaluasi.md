# LOG INTERAKSI LLM - KELOMPOK An Tech A Sync

- **Tanggal Sesi:** 23/10/2026
- **Anggota:** Anggota 3 - Training & Evaluasi
- **Target Modul:** `src/train.py`, `src/evaluate.py`, `src/metrics.py`, checkpoint, dan hasil training Flickr8k
- **LLM yang dipakai:** Codex
- **Tautan Sesi Chat:** [lampirkan tautan/ekspor sesi training dan evaluasi]

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Perencanaan Training)
- **Prompt Kelompok:**
  > "Nah sekarang kan sudah sampai di preprocessing, untuk tahap training selanjutnya gimana berdasarkan README? JANGAN LAKUKAN APAPUN DULU."
- **Alasan Khusus:** Memastikan tahap training mengikuti kontrak arsitektur dan data yang sudah dibuat Anggota 1 dan Anggota 2 sebelum mengubah kode.
- **Output Model:**
  - Menyusun alur `train.py -> uji hafalan 100 gambar -> training penuh -> evaluate.py -> dokumentasi hasil`.
  - Merencanakan Adam, cross-entropy dengan padding diabaikan, gradient clipping, scheduler, early stopping, checkpoint terbaik/terakhir, resume training, dan grafik loss.
  - Menyarankan evaluasi dengan test loss dan metrik caption, bukan accuracy klasifikasi biasa.

### 2. TAHAP IMPLEMENTASI PIPELINE TRAINING DAN EVALUASI
- **Prompt Kelompok (diringkas dari sesi):**
  > "Implementasikan fase training dan evaluasi secara lokal dulu, lengkap dengan test, checkpoint, resume, early stopping, grafik loss, dan mode overfit kecil. Commit hasilnya tanpa menyertakan perubahan notebook yang sudah ada."
- **Alasan Khusus:** Pipeline harus dapat diuji end-to-end di lokal sebelum training penuh, serta tidak boleh mengganggu pekerjaan anggota lain pada notebook.
- **Output Model:**
  - Membuat `src/train.py` untuk training/validation, pemilihan device, reproducibility, checkpoint, resume, scheduler, early stopping, progress batch, grafik loss, dan mode `--overfit-images`.
  - Membuat `src/evaluate.py` untuk memuat checkpoint terbaik, menghitung test loss, dan menyimpan contoh caption.
  - Membuat tes training/checkpoint/evaluasi; seluruh suite awal menghasilkan `21 passed`.
  - Smoke test lokal 1 gambar dan 1 epoch berhasil: data terbaca, training berjalan, checkpoint tersimpan dan dapat dimuat kembali, test loss dihitung, serta caption contoh dihasilkan.
  - Implementasi awal tersimpan pada commit `e1d08c1`; progress per batch ditambahkan pada commit `11b2edc`.

### 3. TAHAP PENANGANAN EROR / DEBUGGING GPU
- **Pesan dari environment lokal:**
  ```text
  torch: 2.14.1+cpu
  torch.version.cuda: None
  torch.cuda.is_available(): False
  ```
- **Prompt Perbaikan ke LLM:**
  > "Laptopku padahal ada GPU-nya kok, RTX."
- **Analisis & Solusi Kelompok:** Windows mendeteksi NVIDIA GeForce RTX 3050 Laptop GPU dengan VRAM 4 GB, tetapi virtual environment memakai PyTorch CPU-only. Solusinya adalah memasang wheel CUDA resmi yang sesuai, lalu memverifikasi device dan operasi tensor di GPU.
- **Prompt Lanjutan:**
  > "Kasih aku command untuk mendownloadnya saja, nanti aku kasih lagi ke kamu kalau sudah selesai."
- **Output Model:**
  ```powershell
  & ".\.venv\Scripts\python.exe" -m pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1 torchvision==0.29.1
  ```
- **Hasil Perbaikan:**
  ```text
  PyTorch       : 2.14.1+cu126
  Torchvision   : 0.29.1+cu126
  CUDA tersedia : True
  GPU           : NVIDIA GeForce RTX 3050 Laptop GPU
  VRAM          : 4 GB
  ```
  Operasi tensor CUDA dan smoke training satu epoch dengan `--device cuda` berhasil. Batch awal dibuat konservatif (`16`, atau `8` jika kehabisan VRAM).

### 4. TAHAP TRAINING PENUH
- **Prompt (diringkas dari sesi):**
  > "Jalankan training penuh 30 epoch menggunakan GPU dan simpan checkpoint serta riwayat loss-nya."
- **Perubahan yang dilakukan:** Training dijalankan pada 6.091 gambar train dan divalidasi pada 1.000 gambar. Riwayat tiap epoch disimpan ke `results/training_history.json`, kurva ke `results/loss_curve.png`, dan model terbaik ke `checkpoints/best_model.pt`.
- **Output Model / Hasil Training:**
  - Training selesai sebanyak **30 epoch** dalam **13.141,14 detik (sekitar 3 jam 39 menit)**.
  - Epoch 1: train loss `3,6851`, validation loss `3,3092`.
  - Epoch terbaik: **28**, validation loss `2,8945`.
  - Epoch 30: train loss `2,7553`, validation loss `2,8993`.
  - Learning rate turun dari `0,001` menjadi `0,0005` mulai epoch 26 setelah validation loss melandai.
  - Artefak hasil training masuk pada commit `eab4799`.

### 5. TAHAP OPTIMASI DAN EVALUASI
- **Prompt:**
  > "Sekarang ada accuracy-nya nggak?"
- **Alasan Khusus:** Image captioning menghasilkan urutan kata, sehingga accuracy klasifikasi tunggal tidak representatif. Evaluasi perlu membandingkan caption prediksi dengan beberapa caption referensi.
- **Perubahan yang dilakukan:** Menambahkan perhitungan BLEU-1 sampai BLEU-4, METEOR, dan CIDEr pada `src/metrics.py`, mengintegrasikannya ke `src/evaluate.py`, menambah tes metrik, dan mendokumentasikan hasil di README.
- **Output Model / Hasil Evaluasi 1.000 Gambar Test:**

  | Metrik | Nilai |
  |---|---:|
  | Test loss | 2,9035 |
  | BLEU-1 | 0,4605 (46,05%) |
  | BLEU-2 | 0,2737 (27,37%) |
  | BLEU-3 | 0,1539 (15,39%) |
  | BLEU-4 | 0,0923 (9,23%) |
  | METEOR | 0,2962 (29,62%) |
  | CIDEr | 0,3093 |

  Evaluasi memakai checkpoint epoch 28. Implementasi metrik dan `results/evaluation.json` tersimpan pada commit `8a146b9`.

### 6. CONTOH OUTPUT CAPTION

Contoh prediksi yang tersimpan di `results/evaluation.json`:

```text
Gambar     : 3018304737_0a46fc5f1d.jpg
Prediksi   : a man in a wetsuit is surfing on a wave
Referensi  : A man catches a huge wave on his surfboard.
             A surfboarder going down a wave.
             A surfer makes a turn on a wave.
             The surfer is riding a wave.
```

Contoh ini menunjukkan model sudah menangkap objek dan aktivitas utama dengan baik. Namun, beberapa contoh lain masih menghasilkan caption yang tidak sesuai gambar, sehingga skor BLEU-4 dan CIDEr masih relatif rendah dan model masih dapat ditingkatkan.

---

## HASIL VALIDASI AKHIR

- Pipeline training, validation, checkpoint, resume, scheduler, early stopping, dan evaluasi berjalan end-to-end.
- Training penuh selesai 30 epoch; checkpoint terbaik berasal dari epoch 28.
- Loss train turun dari `3,6851` menjadi `2,7553`.
- Validation loss terbaik adalah `2,8945`.
- Evaluasi dilakukan pada 1.000 gambar test dengan BLEU, METEOR, dan CIDEr.
- Hasil lengkap tersedia di `results/training_history.json`, `results/loss_curve.png`, dan `results/evaluation.json`.

---

**Commit terkait:** `e1d08c1`, `11b2edc`, `eab4799`, `8a146b9` (branch lama). Setelah dipindah ke branch bersih dan di-merge ke `main` lewat PR #6, hash-nya menjadi `e861b39`, `90222d8`, `508cc76`, `cb1a8ce`, `5365e44`.
