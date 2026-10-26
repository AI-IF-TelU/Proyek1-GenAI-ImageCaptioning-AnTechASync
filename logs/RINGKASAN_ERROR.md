# Ringkasan Error & Solusi - Kelompok An Tech A Sync

Rangkuman seluruh error dan kendala yang muncul selama proyek, dikumpulkan dari log interaksi LLM setiap anggota. Detail lengkap ada di log masing-masing (`logs/anggotaX/`).

| # | Tahap | Error / Kendala | Penyebab | Solusi | Sumber log |
|---|---|---|---|---|---|
| 1 | Setup Colab | `ModuleNotFoundError: No module named 'imp'` saat `%load_ext autoreload` | Colab memakai Python 3.13. Ekstensi autoreload IPython masih mengimpor modul `imp` yang dihapus sejak Python 3.12 | Mengganti autoreload dengan fungsi `muat_ulang()` yang menghapus modul `src` dari `sys.modules` | Anggota 1, 19/10 |
| 2 | Uji model di Colab | `No module named tests.test_encoder` | Colab sudah punya paket lain bernama `tests`, sedangkan folder `tests/` kelompok belum punya `__init__.py` | Menambahkan `tests/__init__.py` | Anggota 1, 19/10 |
| 3 | Git | `! [rejected] ... (fetch first)` | Notebook disimpan dari Colab ke GitHub sehingga branch di GitHub lebih baru | `git pull --rebase` sebelum push | Anggota 1, 19/10 |
| 4 | Uji hafalan model | Hasil `generate` tidak konsisten (perbaikan sebelum uji dijalankan) | Mode `train` saat generate ikut menyalakan Dropout | Memakai `model.generate()` dalam mode `eval` | Anggota 1, 19/10 |
| 5 | Data pipeline (lokal) | `ModuleNotFoundError: No module named 'torch'` | Environment lokal (Python 3.14) belum punya PyTorch/torchvision/pytest | Memasang dependensi lalu menjalankan ulang tes | Anggota 2, 22/10 |
| 6 | Training (lokal) | `torch 2.14.1+cpu`, `cuda.is_available(): False` padahal ada GPU RTX 3050 | PyTorch yang terpasang versi CPU-only | Memasang ulang wheel CUDA 12.6 dengan `--force-reinstall` | Anggota 3, 23/10 |
| 7 | Integrasi Git | Branch `anggota3-training` tidak bisa di-PR ("no common history") | Branch ditimpa salinan dari repo lain sehingga riwayat Git terputus dari `main` | Membuat branch bersih dari `main` dan memindahkan commit tahap 3 (penulis asli dipertahankan) | Anggota 1, 26/10 |
| 8 | Tuning (lokal) | `torch 2.14.1+cpu`, `AssertionError: Torch not compiled with CUDA enabled` di laptop GTX 1660 Ti | Sama dengan #6. Pemasangan pertama dilewati pip karena versi 2.14.1 dianggap sudah ada | `pip install --force-reinstall --no-deps torch==2.14.1+cu126 ...` | Anggota 1, 26/10 |
| 9 | Tuning (lokal) | `RuntimeError: Couldn't open shared file mapping ... error code: <1455>` di epoch 24 | RAM laptop 7,8 GB habis oleh 4 proses pembaca data (`--num-workers 4`) | Melanjutkan dari checkpoint epoch 23 (`--resume`) dengan `--num-workers 1` | Anggota 1, 26/10 |
| 10 | Tuning (Colab) | Training sangat lambat | Membaca dan me-resize ±30.000 gambar per epoch dikerjakan CPU Colab gratis (2 core), sehingga GPU menunggu data | Training dipindah ke GPU laptop | Anggota 1, 26/10 |

## Pola yang dipelajari

1. **Baca baris terakhir traceback dan lokasi file-nya.** Lokasi `dist-packages/...` berarti masalah ada di library atau lingkungan, bukan di kode kelompok (#1).
2. **Lulus di laptop belum tentu lulus di Colab**, begitu pula sebaliknya. Selalu uji di lingkungan target (#2).
3. **Cek versi PyTorch sebelum training.** Akhiran `+cpu` berarti GPU tidak dipakai (#6, #8).
4. **Checkpoint setiap epoch** menyelamatkan training yang terhenti di tengah jalan (#9).
5. **`git pull` sebelum `push`**, dan semua perubahan masuk ke `main` lewat Pull Request (#3, #7).
