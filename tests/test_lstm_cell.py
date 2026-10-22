"""
Uji LSTMCell manual dengan tensor acak.

Jalankan dari root repo:  python -m tests.test_lstm_cell
"""

import torch

from src.lstm_cell import LSTMCell

B, INPUT_SIZE, HIDDEN_SIZE = 4, 256, 512


def test_lstm_cell():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cell = LSTMCell(INPUT_SIZE, HIDDEN_SIZE).to(device)
    x = torch.randn(B, INPUT_SIZE, device=device)
    h = torch.zeros(B, HIDDEN_SIZE, device=device)
    c = torch.zeros(B, HIDDEN_SIZE, device=device)

    # 1. Shape satu langkah
    h1, c1 = cell(x, (h, c))
    print(f"Satu langkah : x {tuple(x.shape)} -> h {tuple(h1.shape)}, c {tuple(c1.shape)}")
    assert h1.shape == (B, HIDDEN_SIZE) and c1.shape == (B, HIDDEN_SIZE)

    # 2. Rumus gate dihitung ulang secara terpisah, hasil harus sama persis
    W, b = cell.gates.weight, cell.gates.bias
    z = torch.cat([x, h], dim=1) @ W.T + b
    zi, zf, zg, zo = z.chunk(4, dim=1)
    c_exp = torch.sigmoid(zf) * c + torch.sigmoid(zi) * torch.tanh(zg)
    h_exp = torch.sigmoid(zo) * torch.tanh(c_exp)
    assert torch.allclose(c1, c_exp, atol=1e-5) and torch.allclose(h1, h_exp, atol=1e-5)
    print("Rumus gate   : sesuai (c' = f*c + i*g, h' = o*tanh(c'))")

    # 3. Bias forget gate = 1, gate lain = 0
    bias = cell.gates.bias.detach()
    assert torch.all(bias[HIDDEN_SIZE : 2 * HIDDEN_SIZE] == 1.0), "Bias forget gate harus 1"
    print("Bias forget  : 1.0 (gate lain dibiarkan acak)")

    # 4. Nilai h selalu di rentang (-1, 1) karena o * tanh(.)
    assert h1.abs().max() < 1.0
    print(f"Rentang h    : [{h1.min().item():.3f}, {h1.max().item():.3f}] (harus di dalam -1..1)")

    # 5. Loop 30 langkah (seperti satu caption) dan gradien mengalir ke langkah pertama
    xs = torch.randn(B, 30, INPUT_SIZE, device=device, requires_grad=True)
    h, c = torch.zeros(B, HIDDEN_SIZE, device=device), torch.zeros(B, HIDDEN_SIZE, device=device)
    for t in range(30):
        h, c = cell(xs[:, t], (h, c))
    h.sum().backward()
    grad_first = xs.grad[:, 0].abs().sum().item()
    assert grad_first > 0, "Gradien tidak sampai ke langkah pertama"
    print(f"30 langkah   : gradien ke kata pertama = {grad_first:.4f} (> 0)")

    n_params = sum(p.numel() for p in cell.parameters())
    print(f"\nJumlah parameter: {n_params:,}")
    print(f"Device: {device}")
    print("LULUS: LSTMCell manual")


if __name__ == "__main__":
    test_lstm_cell()
