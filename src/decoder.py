"""
DecoderLSTM: "mulut" model image captioning.

Menerima fitur gambar dari EncoderCNN (B, 256), lalu menyusun kalimat kata demi kata
memakai LSTMCell manual. Tanpa nn.LSTM bawaan PyTorch.
"""

import torch
import torch.nn as nn

from src.lstm_cell import LSTMCell

# Nomor token khusus (sesuai kontrak di README)
PAD_IDX = 0
START_IDX = 1
END_IDX = 2


class DecoderLSTM(nn.Module):
    """
    Alur tiap langkah waktu t:
        kata ke-t -> Embedding -> LSTMCell (memori h, c) -> Dropout -> Linear -> skor tiap kata di kamus

    Fitur gambar masuk sebagai MEMORI AWAL LSTM:
        h0 = tanh(Linear(fitur)),  c0 = tanh(Linear(fitur))
    Jadi sebelum menulis kata pertama, LSTM sudah "mengingat" isi gambar.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        feature_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Nomor kata -> vektor makna. padding_idx: vektor <pad> selalu nol dan tidak ikut dilatih.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # Fitur gambar -> memori awal LSTM. tanh menjaga nilainya di -1..1, sama seperti h biasa.
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        self.lstm_cell = LSTMCell(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Memori -> skor untuk setiap kata di kamus (logits)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.init_h(features))  # (B, hidden_dim)
        c = torch.tanh(self.init_c(features))  # (B, hidden_dim)
        return h, c

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Training dengan teacher forcing.

        features: (B, feature_dim)
        captions: (B, T) caption utuh, T = 30, diawali <start>
        return  : logits (B, T-1, vocab_size)

        Input  = captions[:, :-1]  (<start>, kata1, ..., kata ke-28)
        Target = captions[:, 1:]   (kata1, kata2, ..., <end>/<pad>)  -> dihitung di train.py
        Logits ke-t adalah tebakan untuk captions[:, t+1].

        Teacher forcing: di setiap langkah LSTM diberi kata yang BENAR dari caption asli,
        bukan tebakannya sendiri. Training jadi lebih cepat dan stabil, karena satu tebakan
        salah tidak merusak langkah-langkah berikutnya.
        """
        inputs = captions[:, :-1]               # (B, T-1)
        embeddings = self.embedding(inputs)     # (B, T-1, embed_dim)

        h, c = self.init_state(features)        # (B, hidden_dim) x2

        outputs = []
        for t in range(inputs.size(1)):
            h, c = self.lstm_cell(embeddings[:, t], (h, c))   # (B, hidden_dim)
            logits_t = self.fc_out(self.dropout(h))           # (B, vocab_size)
            outputs.append(logits_t)

        return torch.stack(outputs, dim=1)      # (B, T-1, vocab_size)

    @torch.no_grad()
    def generate(self, features: torch.Tensor, max_len: int = 30) -> list[int]:
        """
        Membuat caption untuk SATU gambar dengan greedy search.

        features: (1, feature_dim)
        return  : daftar nomor kata, tanpa <start> dan <end>

        Mulai dari <start>. Di tiap langkah ambil kata dengan skor tertinggi, lalu kata itu
        menjadi input langkah berikutnya (tidak ada caption asli, beda dengan teacher forcing).
        Berhenti saat model menghasilkan <end> atau panjangnya mencapai max_len
        (max_len termasuk <start> dan <end>, jadi maksimal max_len - 1 langkah).
        Panggil model.eval() sebelumnya agar Dropout nonaktif.
        """
        assert features.size(0) == 1, "generate hanya untuk satu gambar: features (1, feature_dim)"

        h, c = self.init_state(features)
        word = torch.tensor([START_IDX], device=features.device)  # (1,)

        result = []
        for _ in range(max_len - 1):
            emb = self.embedding(word)                       # (1, embed_dim)
            h, c = self.lstm_cell(emb, (h, c))               # (1, hidden_dim)
            logits = self.fc_out(self.dropout(h))            # (1, vocab_size)
            word = logits.argmax(dim=1)                      # (1,) kata dengan skor tertinggi

            if word.item() == END_IDX:
                break
            result.append(word.item())

        return result
