"""
LSTMCell manual: "memori" decoder.

Ditulis sendiri dari nn.Linear (tanpa nn.LSTM / nn.LSTMCell bawaan PyTorch).
Satu langkah waktu: menerima satu kata (x) dan memori sebelumnya (h, c),
lalu menghasilkan memori baru (h', c').
"""

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """
    Satu sel LSTM. Memorinya ada dua:
    - c (cell state)  : "buku catatan" jangka panjang
    - h (hidden state): keluaran langkah ini, dipakai untuk menebak kata

    Empat bagian yang dihitung tiap langkah:
    - i (input gate)    : seberapa banyak info baru yang DICATAT       -> sigmoid (0..1)
    - f (forget gate)   : seberapa banyak catatan lama yang DISIMPAN   -> sigmoid (0..1)
    - g (candidate)     : ISI info baru yang mungkin dicatat           -> tanh (-1..1)
    - o (output gate)   : seberapa banyak catatan yang DIKELUARKAN     -> sigmoid (0..1)

    Rumus:
        c' = f * c + i * g        (lupakan sebagian catatan lama, tambah info baru)
        h' = o * tanh(c')         (keluarkan sebagian isi catatan)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Satu Linear menghitung keempat gate sekaligus (lebih cepat daripada 4 Linear terpisah).
        # Masukan: gabungan [x, h], keluaran: 4 * hidden_size -> dipotong jadi i, f, g, o.
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        self._init_forget_bias()

    def _init_forget_bias(self):
        # Bias forget gate diisi 1, sehingga di awal training f = sigmoid(~1) ~ 0.73:
        # sel cenderung MENYIMPAN memori, jadi info dari awal kalimat tidak cepat hilang.
        # Urutan potongan: [i, f, g, o] -> forget gate ada di potongan ke-2.
        with torch.no_grad():
            h = self.hidden_size
            self.gates.bias[h : 2 * h].fill_(1.0)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state                                   # (B, hidden_size), (B, hidden_size)

        combined = torch.cat([x, h], dim=1)            # (B, input_size + hidden_size)
        i, f, g, o = self.gates(combined).chunk(4, dim=1)  # masing-masing (B, hidden_size)

        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        g = torch.tanh(g)      # candidate memory
        o = torch.sigmoid(o)   # output gate

        c_next = f * c + i * g             # (B, hidden_size)
        h_next = o * torch.tanh(c_next)    # (B, hidden_size)
        return h_next, c_next
