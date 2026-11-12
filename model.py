"""Arsitektur model dan pemuat checkpoint (.pth) untuk aplikasi klasifikasi Breast Cancer.

Arsitektur di sini HARUS sama dengan kelas BreastCancerMLP pada notebook colab/Training_PyTorch_Colab.ipynb,
karena file .pth hanya menyimpan bobot, bukan definisi kelasnya.
"""
from pathlib import Path

import torch
from torch import nn


class BreastCancerMLP(nn.Module):
    """MLP untuk klasifikasi biner. Keluaran berupa logit (belum sigmoid)."""

    def __init__(self, in_features, hidden=(32, 16), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Predictor:
    """Memuat checkpoint .pth dan melakukan prediksi dari nilai fitur mentah."""

    def __init__(self, path):
        path = Path(path)
        # weights_only=True: hanya memuat tensor dan tipe data dasar (aman dari eksekusi kode di file .pth)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        self.model = BreastCancerMLP(ckpt["in_features"], tuple(ckpt["hidden"]), ckpt["dropout"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.mean = torch.tensor(ckpt["mean"], dtype=torch.float32)
        self.scale = torch.tensor(ckpt["scale"], dtype=torch.float32)
        self.meta = {
            "feature_names": list(ckpt["feature_names"]),
            "feature_min": list(ckpt["feature_min"]),
            "feature_max": list(ckpt["feature_max"]),
            "class_names": list(ckpt["class_names"]),
            "examples": dict(ckpt["examples"]),
            "metrics": dict(ckpt.get("metrics", {})),
        }

    def predict(self, values):
        """values: daftar angka sesuai urutan meta['feature_names']. Mengembalikan peluang kelas benign (1)."""
        x = torch.tensor([values], dtype=torch.float32)
        x = (x - self.mean) / self.scale
        with torch.no_grad():
            p_benign = torch.sigmoid(self.model(x)).item()
        return p_benign
