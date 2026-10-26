"""
Aplikasi web sederhana: upload foto, caption langsung muncul.

Jalankan dari root repo:
    streamlit run app.py

Checkpoint (best_model.pt) tidak disimpan di GitHub. Aplikasi mencari otomatis di:
    1. variabel lingkungan CAPTION_CHECKPOINT
    2. checkpoints/**/best_model.pt di folder repo
    3. C:/Users/<user>/ic_data/runs/**/best_model.pt
atau isi path-nya manual di sidebar.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.dataset import build_eval_transform
from src.predict import load_model
from src.train import resolve_device

REPO_ROOT = Path(__file__).resolve().parent


def find_checkpoints() -> list[Path]:
    found: list[Path] = []
    env = os.environ.get("CAPTION_CHECKPOINT")
    if env and Path(env).is_file():
        found.append(Path(env))
    for root in (REPO_ROOT / "checkpoints", Path.home() / "ic_data" / "runs"):
        if root.is_dir():
            found.extend(sorted(root.rglob("best_model.pt")))
    return list(dict.fromkeys(found))  # buang duplikat, urutan tetap


@st.cache_resource(show_spinner="Memuat model...")
def get_model(checkpoint_path: str):
    device = resolve_device("auto")
    model, vocab, max_len = load_model(checkpoint_path, device)
    return model, vocab, max_len, device


@torch.no_grad()
def make_caption(image: Image.Image, checkpoint_path: str) -> str:
    model, vocab, max_len, device = get_model(checkpoint_path)
    tensor = build_eval_transform()(image.convert("RGB")).unsqueeze(0).to(device)  # (1, 3, 128, 128)
    return model.caption_image(tensor, vocab, max_len=max_len)


st.set_page_config(page_title="Image Captioning", page_icon="🖼️", layout="wide")
st.title("🖼️ Image Captioning from Scratch")
st.caption("Encoder Vanilla CNN + Decoder LSTM manual, dilatih dari nol pada Flickr8k. Upload foto, caption langsung muncul.")

with st.sidebar:
    st.header("Model")
    options = find_checkpoints()
    if options:
        labels = [str(p.relative_to(p.parents[2])) if len(p.parents) > 2 else str(p) for p in options]
        choice = st.selectbox("Checkpoint", range(len(options)), format_func=lambda i: labels[i])
        checkpoint = str(options[choice])
    else:
        checkpoint = ""
    manual = st.text_input("Atau isi path best_model.pt", value="")
    if manual.strip():
        checkpoint = manual.strip().strip('"')

    if checkpoint and Path(checkpoint).is_file():
        model, vocab, _, device = get_model(checkpoint)
        st.success(f"Model siap ({device.type.upper()})")
        st.write(f"Kosakata: {len(vocab):,} kata")
        st.write(f"Parameter: {sum(p.numel() for p in model.parameters()):,}")
    else:
        st.error("Checkpoint belum ditemukan. Isi path best_model.pt di atas.")
        st.stop()

files = st.file_uploader(
    "Upload satu atau beberapa foto", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True
)

if not files:
    st.info("Belum ada foto. Seret foto ke kotak di atas atau klik **Browse files**.")
else:
    cols = st.columns(3)  # selalu 3 kolom agar 1 foto tidak tampil terlalu besar
    for i, file in enumerate(files):
        image = Image.open(file)
        caption = make_caption(image, checkpoint)
        with cols[i % len(cols)]:
            st.image(image, width="stretch")
            st.markdown(f"**📝 {caption}**")

st.divider()
st.caption(
    "Catatan: model dilatih dari nol (tanpa pretrained) pada ±6.000 gambar Flickr8k berukuran 128×128, "
    "jadi caption untuk adegan yang jarang muncul di data bisa kurang tepat. "
    "`<unk>` berarti kata yang tidak ada di kosakata model."
)
