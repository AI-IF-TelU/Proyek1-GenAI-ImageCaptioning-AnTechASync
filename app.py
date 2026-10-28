"""
Aplikasi web sederhana: upload foto, caption langsung muncul.

Jalankan dari root repo:
    streamlit run app.py

Versi online: Streamlit Community Cloud memakai models/caption_model_exp02.pt (bobot model
ringkas yang ikut disimpan di GitHub). Aplikasi mencari model otomatis di:
    1. variabel lingkungan CAPTION_CHECKPOINT
    2. models/*.pt di folder repo
    3. checkpoints/**/best_model.pt di folder repo
    4. C:/Users/<user>/ic_data/runs/**/best_model.pt
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.dataset import build_eval_transform
from src.predict import load_model
from src.train import resolve_device

REPO_ROOT = Path(__file__).resolve().parent


def find_checkpoint() -> Path | None:
    found: list[Path] = []
    env = os.environ.get("CAPTION_CHECKPOINT")
    if env and Path(env).is_file():
        found.append(Path(env))
    if (REPO_ROOT / "models").is_dir():
        found.extend(sorted((REPO_ROOT / "models").glob("*.pt")))
    for root in (REPO_ROOT / "checkpoints", Path.home() / "ic_data" / "runs"):
        if root.is_dir():
            found.extend(sorted(root.rglob("best_model.pt")))
    return found[0] if found else None


@st.cache_resource(show_spinner=False)
def get_model(checkpoint_path: str):
    device = resolve_device("auto")
    model, vocab, max_len = load_model(checkpoint_path, device)
    return model, vocab, max_len, device


@torch.no_grad()
def make_caption(image: Image.Image, checkpoint_path: str) -> str:
    model, vocab, max_len, device = get_model(checkpoint_path)
    tensor = build_eval_transform()(image.convert("RGB")).unsqueeze(0).to(device)  # (1, 3, 128, 128)
    return model.caption_image(tensor, vocab, max_len=max_len)


def photo_card(image: Image.Image, caption: str) -> str:
    """Foto berbingkai putih dengan caption bergaya mesin ketik di bawahnya."""
    preview = image.convert("RGB")
    preview.thumbnail((1000, 1000))
    buffer = io.BytesIO()
    preview.save(buffer, "JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return (
        f'<div class="kartu"><img src="data:image/jpeg;base64,{encoded}" alt="{caption}">'
        f'<p class="cap">{caption}</p></div>'
    )


st.set_page_config(page_title="Image Captioning", page_icon="🖼️", layout="centered", initial_sidebar_state="collapsed")

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=Courier+Prime&display=swap" rel="stylesheet">
<style>
.stApp { background:#FAFAF7; }
header, footer, [data-testid="stSidebar"], [data-testid="stToolbar"] { display:none !important; }
.block-container { max-width:860px; padding-top:4.5rem; }
html, body, .stApp, p, label, div { color:#1F1F1F; font-family:'EB Garamond', serif; }
.stMarkdown p.judul { font-family:'EB Garamond', serif !important; font-style:italic; font-size:56px !important;
    text-align:center; line-height:1.1 !important; margin:0; }
.stMarkdown p.kelompok { text-align:center; font-size:20px !important; opacity:.7; margin:14px 0 40px 0; letter-spacing:.5px; }
.kartu { background:#FFFFFF; padding:10px; border:1px solid #DDDAD2; margin:0 auto 28px auto; }
.kartu img { width:100% !important; height:auto !important; max-width:none !important; object-fit:cover !important; display:block; }
.stMarkdown p.cap { font-family:'Courier Prime', monospace !important; font-size:17px !important; color:#3A3A3A !important;
    text-align:center; margin:14px 6px 6px 6px; line-height:1.35 !important; }
[data-testid="stFileUploaderDropzone"] { background:transparent !important; border:1.5px dashed #1F1F1F55 !important; border-radius:4px; }
[data-testid="stFileUploaderDropzone"] * { color:#1F1F1F !important; }
[data-testid="stFileUploaderDropzone"] button { background:transparent !important; border:1px solid #1F1F1F55 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { display:none !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="judul">Image Captioning</p><p class="kelompok">An Tech A Sync</p>', unsafe_allow_html=True)

checkpoint = find_checkpoint()
if checkpoint is None:
    st.error("Model belum ditemukan. Letakkan file .pt di folder models/.")
    st.stop()

files = st.file_uploader(
    "Upload foto", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, label_visibility="collapsed"
)

if files:
    columns = st.columns(min(3, len(files)), gap="large")
    for index, file in enumerate(files):
        image = Image.open(file)
        with columns[index % len(columns)]:
            st.markdown(photo_card(image, make_caption(image, str(checkpoint))), unsafe_allow_html=True)
